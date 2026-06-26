import os,warnings,logging
import sqlite3
import json
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import base64
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.agents.agent import AgentExecutor
from langchain_classic.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
import uuid
from typing import List

warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)
load_dotenv()

root_dir = os.path.join(os.getcwd(), 'data')
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
if not os.path.exists(os.path.join(root_dir, "models")):
    os.makedirs(os.path.join(root_dir, "models"))
db_path = os.path.join(root_dir, 'memory.db')

st.set_page_config(page_title="Faberlic Assistant", page_icon="🛍️")
baza_yolu = os.path.join(root_dir, "chroma_db")
excel_fayl_adi = 'Faberlic_Bazasi.xlsx'
@st.cache_data 
def load_excel_data():
    return pd.read_excel(excel_fayl_adi)
df=load_excel_data()
@st.cache_resource
def get_chroma_collection():
    embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=baza_yolu)
    return client.get_or_create_collection(name='data_db', embedding_function=embedding_model)

collection = get_chroma_collection()
baza_movcuddur = os.path.exists(baza_yolu) and len(os.listdir(baza_yolu)) > 0
    
if not baza_movcuddur:
            st.info("🔄 Vektor bazası tapılmadı. Excel faylı ilk dəfə emal edilir, zəhmət olmasa gözləyin...")
            
            documents = []
            metadatas = []
            ids = []
            
            for index, row in df.iterrows():
                setir_metni = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                
                if setir_metni.strip() == "":
                    continue
                
                documents.append(setir_metni)
                metadatas.append({"row_number": index + 1})
                ids.append(f"id_{index}")
            

            if documents:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                st.success(f"✨ {len(documents)} sətir uğurla vektorlaşdırıldı və diskə qeyd olundu!")
        
else:
    st.sidebar.success("📊 Lokal Faberlic Vektor Bazası aktivdir!")

class Gelivy:
    def __init__(self, user_uuid=None):
        self.user_uuid = user_uuid
        if "files" not in st.session_state:
            st.session_state.files={}
        self.api_key=os.getenv("API_KEYS")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_uuid TEXT UNIQUE,
                    messages TEXT,
                    tarix DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        self.setup_agent()

    def load_chat_history(self):
        if not self.user_uuid:
            return []
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT messages FROM history WHERE user_uuid = ?", (self.user_uuid,))
            row = cursor.fetchone()

        langchain_messages = []
        if row and row[0]:
            saved_messages = json.loads(row[0])
            for msg in saved_messages:
                if msg["type"] == "human":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["type"] == "ai":
                    langchain_messages.append(AIMessage(content=msg["content"]))
        return langchain_messages

    def save_chat_history(self, langchain_messages):
        if not self.user_uuid:
            return
        serializable_messages = []
        for msg in langchain_messages:
            if isinstance(msg, HumanMessage):
                serializable_messages.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                serializable_messages.append({"type": "ai", "content": msg.content})

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO history (user_uuid, messages) VALUES (?, ?)", 
                            (self.user_uuid, json.dumps(serializable_messages)))
            conn.commit()
    
    def setup_agent(self):
        self.llm = ChatGroq(
            api_key=self.api_key,
            model="openai/gpt-oss-120b",
            temperature=0.5,
            max_retries=3
        )
        
        self.tools = [self.generate_video,
                      self.get_vision_tool(),
                      self.get_available_images_info
                      ]
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Sənin adın Gelivy-dır. Sən Faberlic ekosisteminin qlobal, vizyoner və hər kəsi kəşf etməyə ruhlandıran Baş Mentorusan. Qarşındakı istifadəçiyə gələcəyin Top Lideri və real sahibkar kimi yanaşırsan.

Uğur Strategiyan:
- **Kopyalana Bilən (Duplicable) Metodlar:** Verdiyin hər bir biznes məsləhəti, rəqəmsal satış taktikası və ya TikTok/Reels ssenarisi elə sadə və effektiv olmalıdır ki, hər kəs tərəfindən dərhal icra edilə bilsin.
- **Vizyon və Dərin Analitika:** Yeni başlayanlara biznesin qlobal miqyasını göstər, böyük liderlərə isə strukturlarını qorumaq, 'Yan Həcm' balansını idarə etmək və yeni brilyant liderlər yetişdirmək üçün strateji yollar təklif et.
- **Struktur, Aydınlıq və Enerji:** Şəbəkə marketinqi terminlərini (ŞB, QH, Marketinq Planı) hər kəsin anlayacağı şəkildə, dürüst istifadə et. Uzun paraqraflar yazma, vacib məlumatları **qalın şriftlə** və bənd-bənd təqdim et. Danışıq tonun hər zaman yüksək motivasiyalı, enerjili, peşəkar və səmimi olmalıdır.

    Məlumat üçün KONTEKS-dən istifadə et:
#KONTEKS:
{context}"""),       
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
    @tool
    @staticmethod
    def get_available_images_info():
        """Sistemdə mövcud olan şəkillərin adlarını və qısa təsvirlərini qaytarır.
        Model hansı şəkli seçəcəyinə qərar vermək üçün bu alətdən istifadə etməlidir.
        """
        info_text = "Available images in the system:\n"
        
        for name , data in st.session_state.files.items():
            info_text += f"- Name: '{name}' | Description: {data}\n"
        return info_text
    
    @staticmethod
    def describe_image_with_vision(images:List[str],prompt:str):
        try:
            st.write("Vision modeli yüklənir...")
            result_text=""
            info_text=""
            llm_vision = ChatGroq(
            api_key=os.getenv("API_KEYS"),
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.5
            )
            
            for img in images:
                if hasattr(img, "name"):
                    file_name = img.name
                    image_bytes = img.getvalue()
                # Variant B: Əgər Agent alətdən sadəcə faylın adını (string) göndəribsə
                elif isinstance(img, str):
                    file_name = img
                    if "image_cache" in st.session_state and file_name in st.session_state.image_cache:
                        image_bytes = st.session_state.image_cache[file_name]
                    elif os.path.exists(file_name):
                        with open(file_name, "rb") as f:
                            image_bytes = f.read()
                    else:
                        st.error(f"❌ '{file_name}' şəklinin baytları keşdə tapılmadı!")
                        continue
                else:
                    continue
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text", 
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                }
            }
        ]
    }
]
        
                response =llm_vision.invoke(
                    input=messages
                )
                result_text=response.content.strip()
                file_name = img.name if hasattr(img, "name") else str(img)
                st.session_state.files[file_name] = result_text
                info_text += f"Name: {file_name} | Description: {result_text}\n"
            return info_text
        except Exception as e:
            st.error(f"Vision Xətası: {str(e)}")

    @staticmethod
    def get_vision_tool():
        @tool
        def image_analyzer_tool(images:List[str],prompt:List[str]) -> str:
            """İstifadəçinin istəyinə görə verilmiş şəkli analiz edir.
        Args:
            image - FİLES lüğətindən istifadəçiyə lazım olan fayl və ya fayllar seçilməlidir.  
            prompt - İstifadəçinin sualına əsasən İngiliscə yazılmalıdır.
        """
            return Gelivy.describe_image_with_vision(images=images, prompt=prompt)
        return image_analyzer_tool
    
    @tool
    @staticmethod
    def generate_video(prompt: str) -> str:
        """İstifadəçinin yüklədiyi şəkli və daxil etdiyi mətni (prompt) birləşdirərək animasiya yaradır.
    
    Bu funksiya başlanğıc nöqtəsi olaraq st.session_state daxilindəki şəkli əsas götürür. 
    Giriş parametru olan 'prompt' dəyişəni isə şəkildəki obyektlərin necə hərəkət edəcəyini, 
    fiziki dinamikanı, kadrda baş verəcək dəyişiklikləri və kamera hərəkətlərini (məs. yaxınlaşma, 
    fırlanma) neyron şəbəkəsinə diktə etmək üçün istifadə olunur. Model bu mətni analiz edərək 
    şəkil üzərində ardıcıl kadrlar (frames) generasiya edir və sonda onları MP4 videosuna çevirir."""
        return ""
    def ask(self, question, collection=None, image_description=""):
            try:
                full_context = ""
                if image_description:
                    full_context += f"【ŞƏKLİNİN TƏSVİRİ:】\n{image_description}\n\n"
                
                if collection is not None:
                    query_results = collection.query(query_texts=[question], n_results=3)
                    if query_results and 'documents' in query_results and query_results['documents'] and query_results['documents'][0]:
                        full_context += "【MƏLUMAT BAZASI KONTEKSİ:】\n" + "\n".join(query_results['documents'][0])
                
                if not full_context:
                    full_context = 'Məlumat və ya aktiv şəkil yoxdur...'
                
                history = self.load_chat_history()
                response = self.agent_executor.invoke({
                    "input": question,
                    "context": full_context,
                    "chat_history": history
                })
                
                history.append(HumanMessage(content=question))
                history.append(AIMessage(content=response['output']))
                self.save_chat_history(history)

                return response['output']
            except Exception as e:
                error_msg = str(e)
                return f"Error: {error_msg}"

cookies = st.context.cookies
user_full_name = cookies.get("faberlic_user_name")
user_uuid = cookies.get("faberlic_user_uuid")
if "bot" not in st.session_state:
    st.session_state.bot=Gelivy(user_uuid=user_uuid)
bot=st.session_state.bot
st.session_state.file_infos = "There is no information..."
if 'messages' not in st.session_state:
    st.session_state.messages = []

if not user_full_name or not user_uuid:
    st.title("🛍️ GelivyAI Xoş Gəldiniz!")
    st.write("Sistemi istifadə etmək üçün zəhmət olmasa qeydiyyat formunu doldurun.")

    with st.form("qeydiyyat_formu"):
        col1, col2 = st.columns(2)
        with col1:
            ad = st.text_input("Adınız", placeholder="Məs: Orxan")
        with col2:
            soyad = st.text_input("Soyadınız", placeholder="Məs: Əliyev")
        raziliq = st.checkbox("Məlumatlarımın sistemdə saxlanılmasına razıyam.")
        
        submit_button = st.form_submit_button("Sistemi Başlat 🚀")

        if submit_button:
            if ad and soyad and raziliq:
                tam_ad = f"{ad} {soyad}"
                yeni_id = str(uuid.uuid4())[:8] 
                
                st.components.v1.html(f"""
                    <script>
                    document.cookie = "faberlic_user_name={tam_ad}; path=/; max-age=31536000;";
                    document.cookie = "faberlic_user_uuid={yeni_id}; path=/; max-age=31536000;";
                    window.parent.location.reload();
                    </script>
                """, height=0)
                st.success("Giriş edilir...")
            else:
                st.error("Zəhmət olmasa bütün xanaları doldurun və razılığı təsdiqləyin.")
else:
    st.sidebar.title(f"👤 {user_full_name}")
    st.sidebar.write(f"🆔 ID: {user_uuid}")
    
    if st.sidebar.button("🔴 Məni Unut (Çıxış)"):
        st.components.v1.html("""
            <script>
            document.cookie = "faberlic_user_name=; path=/; expires=Thu, 01 Jan 1970 00:00:01 GMT;";
            document.cookie = "faberlic_user_uuid=; path=/; expires=Thu, 01 Jan 1970 00:00:01 GMT;";
            window.parent.location.reload();
            </script>
        """, height=0)
        st.session_state.messages = []
        st.rerun()

    st.title('Gelivy Ai')

    history_messages = bot.load_chat_history()
    if history_messages and not st.session_state.messages:
        for msg in history_messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            st.session_state.messages.append({"role": role, "content": msg.content, "type": "text"})

    for msg in st.session_state.messages:
        avatar_icon = '👨‍💻' if msg['role'] == 'user' else '🤖'
        with st.chat_message(msg['role'], avatar=avatar_icon):
            if msg.get("type") == "video":
                st.video(msg['content'])
            else:
                st.markdown(msg['content'])

    with st.sidebar:
        st.subheader("🎬 Şəkil Yükləmə")
        uploaded_files = st.file_uploader('Şəkil yükləyin (.png, .jpg, .jpeg, .webp)', 
                                         type=['png', 'jpg', 'jpeg', 'webp'],
                                         accept_multiple_files=True)
        
        if uploaded_files:
            if "image_cache" not in st.session_state:
                st.session_state.image_cache = {}
                
            new_files = []
            for f in uploaded_files:
                st.session_state.image_cache[f.name] = f.getvalue()
                if f.name not in st.session_state.files:
                    new_files.append(f)
    
            if new_files:
                with st.spinner("Yeni şəkillər analiz edilir..."):
                    bot.describe_image_with_vision(images=new_files, prompt="Please describe the main subject of this image. Explain what it is and write a short summary of 2 to 3 sentences.")
    
            info_text = "Available images in the system:\n"
            for name, desc in st.session_state.files.items():
                info_text += f"- Name: '{name}' | Description: {desc}\n"
            st.session_state.file_infos = info_text
        else:
            st.session_state.files = {}
            st.session_state.image_cache = {} 
            st.session_state.file_infos = "There is no information..."
    if user_input := st.chat_input("Mesajınızı yazın..."):
        with st.chat_message("user", avatar='👨‍💻'):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input, "type": "text"})

        if "last_generated_video" in st.session_state:
            del st.session_state.last_generated_video

        with st.chat_message("assistant", avatar='🤖'):
            with st.spinner("Gelivy..."):
                if st.session_state.file_infos:
                    img_desc=st.session_state.get("file_infos")
                else:
                    img_desc=None
                bot_response = bot.ask(question=user_input, collection=collection, image_description=img_desc)
                
                if "last_generated_video" in st.session_state:
                    st.video(st.session_state.last_generated_video)
                    st.session_state.messages.append({"role": "assistant", "content": st.session_state.last_generated_video, "type": "video"})
                
                st.markdown(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response, "type": "text"})
