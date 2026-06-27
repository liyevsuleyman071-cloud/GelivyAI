import os, warnings, logging
import psycopg2
import json
import base64
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.agents.agent import AgentExecutor
from langchain_classic.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from typing import List
import random
import smtplib
from email.mime.text import MIMEText
from streamlit_js_eval import streamlit_js_eval


warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)

st.set_page_config(page_title="Faberlic Assistant", page_icon="🛍️")
collection=None

class Gelivy:
    def __init__(self, user_uuid=None):
        self.user_uuid = user_uuid
        if "files" not in st.session_state:
            st.session_state.files = {}
        self.api_key = st.secrets["API_KEY"]
        with psycopg2.connect(st.secrets["DB_URL"]) as conn:            
            with conn.cursor() as cursor:                                      
                cursor.execute("""                                          
                    CREATE TABLE IF NOT EXISTS history (                    
                        id SERIAL PRIMARY KEY,               
                        user_uuid TEXT UNIQUE,                              
                        messages TEXT,
                        tarix TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )                                                      
                """)   
                cursor.execute("""                                          
                    CREATE TABLE IF NOT EXISTS rag_data (                    
                        id SERIAL PRIMARY KEY,               
                        content TEXT,                              
                        tarix TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )                                                      
                """)
                conn.commit()
        self.setup_agent()

    def load_chat_history(self):
        if not self.user_uuid:
            return []
        with psycopg2.connect(st.secrets["DB_URL"]) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT messages FROM history WHERE user_uuid = %s", (self.user_uuid,))
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

        with psycopg2.connect(st.secrets["DB_URL"]) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO history (user_uuid, messages) 
                    VALUES (%s, %s)
                    ON CONFLICT (user_uuid) 
                    DO UPDATE SET messages = EXCLUDED.messages
                """, (self.user_uuid, json.dumps(serializable_messages)))
                conn.commit()
    
    def setup_agent(self):
        self.llm = ChatGroq(
            api_key=self.api_key,
            model="openai/gpt-oss-120b",
            temperature=0.3,
            max_retries=3,
            max_tokens=2048
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
        for name, data in st.session_state.files.items():
            info_text += f"- Name: '{name}' | Description: {data}\n"
        return info_text
    
    @staticmethod
    def describe_image_with_vision(images: List[str], prompt: str):
        try:
            st.write("Vision modeli yüklənir...")
            result_text = ""
            info_text = ""
            llm_vision = ChatGroq(
                api_key=st.secrets["API_KEY"],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.3
            )
            
            for img in images:
                if hasattr(img, "name"):
                    file_name = img.name
                    image_bytes = img.getvalue()
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
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                        ]
                    }
                ]
        
                response = llm_vision.invoke(input=messages)
                result_text = response.content.strip()
                file_name = img.name if hasattr(img, "name") else str(img)
                st.session_state.files[file_name] = result_text
                info_text += f"Name: {file_name} | Description: {result_text}\n"
            return info_text
        except Exception as e:
            st.error(f"Vision Xətası: {str(e)}")

    @staticmethod
    def get_vision_tool():
        @tool
        def image_analyzer_tool(images: List[str], prompt: List[str]) -> str:
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
        """İstifadəçinin yüklədiyi şəkli və daxil etdiyi mətni (prompt) birləşdirərək animasiya yaradır."""
        return ""

    def ask(self, question, image_description=""):
        try:
            full_context = ""
            if image_description:
                full_context += f"【ŞƏKLİNİN TƏSVİRİ:】\n{image_description}\n\n"
            
            # Axtarış üçün təmiz kiçik hərflərlə sorğu və faizli SQL sorğusu hazırlayırıq
            clean_question = question.strip()
            search_query = f"%{clean_question}%" 
        
            with psycopg2.connect(st.secrets["DB_URL"]) as conn:
                with conn.cursor() as cursor:
                    # 1. Standart RAG Məlumat bazası
                    cursor.execute(
                        "SELECT content FROM rag_data WHERE content ILIKE %s LIMIT 5", 
                        (search_query,)
                    )
                    rag_rows = cursor.fetchall()
                
                    if rag_rows:
                        full_context += "【MƏLUMAT BAZASI KONTEKSİ:】\n"
                        for row in rag_rows:
                            full_context += f"{row[0]}\n"
                        full_context += "\n"

                    # 2. Faberlic Məhsul Bazası (AĞILLI VƏ DƏQİQ SQL SORĞUSU)
                    # %s (yəni istifadəçinin sualı) daxilində artikul kodunun keçib-keçmədiyi yoxlanılır
                    product_sql = """
                        SELECT artikul, mehsul_adi, aciqlama, ilkin_qiymet, kataloq_qiymeti, anbar_qiymeti, saticilara_ozel_endirimli_qiymet, mehsul_bali 
                        FROM faberlic_mehsullar 
                        WHERE %s ILIKE CONCAT('%%', artikul, '%%') 
                        OR mehsul_adi ILIKE %s
                        LIMIT 5
                    """
                    # Diqqət: SQL daxilində CONCAT istifadə etdik ki, cümlə daxilindəki artikulu tuta bilsin
                    cursor.execute(product_sql, (clean_question, search_query))
                    product_rows = cursor.fetchall()
                
                    if product_rows:
                        full_context += "【TAPILAN FABERLIC MƏHSULLARI:】\n"
                        for row in product_rows:
                            artikul, ad, aciqlama, ilkin, kataloq, anbar, promo, bal = row
                            full_context += f"- Məhsul: {ad} (Artikul: {artikul})\n"
                            full_context += f"  Açıqlama: {aciqlama}\n"
                            if ilkin: full_context += f"  İlkin Qiymət (Üstü xəttli): {ilkin} ₼\n"
                            if kataloq: full_context += f"  Kataloq Qiyməti: {kataloq} ₼\n"
                            if anbar: full_context += f"  Anbar Qiyməti: {anbar} ₼\n"
                            if promo: full_context += f"  Aksiya/Satıcı Qiyməti: {promo} ₼\n"
                            if bal: full_context += f"  Qazanılacaq Bal: {bal}\n"
                            full_context += "-----------------------------------\n"
                            
            if not full_context.strip() or full_context == f"【ŞƏKLİNİN TƏSVİRİ:】\n{image_description}\n\n":
                full_context += "Məlumat bazasında bu məhsula və ya suala uyğun aktiv məlumat tapılmadı..."
            
            history = self.load_chat_history()
        
            # Modelə məlumatları və yenilənmiş təmiz KONTEKS-i göndəririk
            response = self.agent_executor.invoke({
                "input": question,
                "context": full_context,
                "chat_history": history
            })
        
            # Keçmişi yeniləyirik
            from langchain_core.messages import HumanMessage, AIMessage
            history.append(HumanMessage(content=question))
            history.append(AIMessage(content=response['output']))
            self.save_chat_history(history)

            return response['output']
        
        except Exception as e:
            return f"Error: {str(e)}"

user_agent = streamlit_js_eval(js_expressions="navigator.userAgent", key="UA")
screen_width = streamlit_js_eval(js_expressions="screen.width", key="SW")

if not user_agent or not screen_width:
    st.info("Sistem yüklənir, zəhmət olmasa gözləyin...")
    st.stop()

cihaz_iz = f"{user_agent}_{screen_width}"

def email_kod_gonder(alici_email, kod):
    # ⚠️ Gmail daxilindən aldığın 16 rəqəmli App Password-u bura yazmalısan
    GÖNDƏRƏN_EMAIL = st.secrets.get("GÖNDƏRƏN_EMAIL", "sənin_gelivy_poçtun@gmail.com")
    GÖNDƏRƏN_ŞİFRƏ = st.secrets.get("GÖNDƏRƏN_ŞİFRƏ", "gmail_alınan_16_rəqəmli_app_password") 

    SÖZÜMÜZ = f"Gelivy AI qeydiyyat təsdiq kodunuz: {kod}\n\nZəhmət olmasa bu kodu heç kimlə paylaşmayın."
    msg = MIMEText(SÖZÜMÜZ, 'plain', 'utf-8')
    msg['Subject'] = 'Gelivy AI - OTP Doğrulama'
    msg['From'] = GÖNDƏRƏN_EMAIL
    msg['To'] = alici_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GÖNDƏRƏN_EMAIL, GÖNDƏRƏN_ŞİFRƏ)
            server.sendmail(GÖNDƏRƏN_EMAIL, alici_email, msg.as_string())
        return True
    except Exception:
        return False

with psycopg2.connect(st.secrets["DB_URL"]) as conn:
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS istifadeciler (
                id SERIAL PRIMARY KEY,
                ad_soyad TEXT,
                email TEXT UNIQUE,
                sifre TEXT,
                cihaz_hash TEXT,
                tarix TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()

user_full_name = st.query_params.get("faberlic_user_name")
user_uuid = st.query_params.get("faberlic_user_uuid")

if "bot" not in st.session_state:
    st.session_state.bot = Gelivy(user_uuid=user_uuid)
bot = st.session_state.bot
st.session_state.file_infos = "There is no information..."

if 'messages' not in st.session_state:
    st.session_state.messages = []


if "otp_kodu" not in st.session_state: st.session_state.otp_kodu = None
if "otp_gonderildi" not in st.session_state: st.session_state.otp_gonderildi = False
if "mveqqeti_qeydiyyat_data" not in st.session_state: st.session_state.mveqqeti_qeydiyyat_data = {}

if not user_full_name or not user_uuid:
    st.title("🛍️ GelivyAI Xoş Gəldiniz!")
    st.write("Sistemi istifadə etmək üçün daxil olun və ya qeydiyyatdan keçin.")

    tab_giris, tab_qeydiyyat = st.tabs(["🔐 Giriş Et", "📝 Qeydiyyat Ol"])

    with tab_giris:
        with st.form("giris_formu"):
            st.subheader("Hesabınıza Giriş Edin")
            giris_email = st.text_input("E-poçt ünvanınız", key="g_email").strip().lower()
            giris_sifre = st.text_input("Şifrəniz", type="password", key="g_sifre")
            giris_submit = st.form_submit_button("Giriş Et 🚀")
            
            if giris_submit:
                if giris_email and giris_sifre:
                    with psycopg2.connect(st.secrets["DB_URL"]) as conn:
                        with conn.cursor() as cursor:
                            cursor.execute(
                                "SELECT ad_soyad, id FROM istifadeciler WHERE email = %s AND sifre = %s",
                                (giris_email, giris_sifre)
                            )
                            user = cursor.fetchone()
                    
                    if user:
                        st.query_params["faberlic_user_name"] = user[0]
                        st.query_params["faberlic_user_uuid"] = str(user[1])
                        st.success(f"🎉 Xoş gəldiniz, {user[0]}! Giriş edilir...")
                        st.rerun()
                    else:
                        st.error("❌ E-poçt və ya şifrə yanlışdır!")
                else:
                    st.error("Zəhmət olmasa bütün sahələri doldurun.")

    with tab_qeydiyyat:
        if not st.session_state.otp_gonderildi:
            with psycopg2.connect(st.secrets["DB_URL"]) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT ad_soyad FROM istifadeciler WHERE cihaz_hash = %s", (cihaz_iz,))
                    cihaz_varmi = cursor.fetchone()
                    
            if cihaz_varmi:
                st.error(f"⚠️ Bu cihazdan artıq bir hesab ({cihaz_varmi[0]}) qeydiyyatdan keçib. İkinci hesaba icazə verilmir!")
            else:
                with st.form("qeydiyyat_formu"):
                    st.subheader("Yeni Hesab Yaradın")
                    col1, col2 = st.columns(2)
                    with col1:
                        q_ad = st.text_input("Adınız", placeholder="Məs: Orxan")
                    with col2:
                        q_soyad = st.text_input("Soyadınız", placeholder="Məs: Əliyev")
                    
                    q_email = st.text_input("E-poçt (Gmail, Mail.ru və s.)").strip().lower()
                    q_sifre = st.text_input("Şifrə təyin edin", type="password")
                    q_sifre_tekrar = st.text_input("Şifrəni yenidən yazın", type="password")
                    raziliq = st.checkbox("Məlumatlarımın sistemdə saxlanılmasına razıyam.", key="q_raziliq")
                    
                    kod_gonder_btn = st.form_submit_button("Təsdiq Kodunu Gönder ✉️")
                    
                    if kod_gonder_btn:
                        if q_ad and q_soyad and q_email and q_sifre and q_sifre_tekrar and raziliq:
                            if q_sifre != q_sifre_tekrar:
                                st.error("❌ Daxil etdiyiniz şifrələr üst-üstə düşmür!")
                            else:
                                # E-poçtun dublikat olub-olmadığını yoxlayırıq
                                with psycopg2.connect(st.secrets["DB_URL"]) as conn:
                                    with conn.cursor() as cursor:
                                        cursor.execute("SELECT id FROM istifadeciler WHERE email = %s", (q_email,))
                                        email_varmi = cursor.fetchone()
                                        
                                if email_varmi:
                                    st.error("❌ Bu e-poçt ünvanı artıq qeydiyyatdan keçib!")
                                else:
                                    # Kod formalaşdırılır və göndərilir
                                    random_otp = str(random.randint(100000, 999999))
                                    st.session_state.otp_kodu = random_otp
                                    st.session_state.mveqqeti_qeydiyyat_data = {
                                        "ad_soyad": f"{q_ad} {q_soyad}",
                                        "email": q_email,
                                        "sifre": q_sifre
                                    }
                                    
                                    with st.spinner("Təsdiq kodu poçtunuza göndərilir..."):
                                        if email_kod_gonder(q_email, random_otp):
                                            st.session_state.otp_gonderildi = True
                                            st.rerun()
                                        else:
                                            st.error("Kodu göndərmək mümkün olmadı. E-poçtu düzgün yazdığınızdan əmin olun.")
                        else:
                            st.error("Zəhmət olmasa bütün xanaları doldurun və razılığı təsdiqləyin.")
        else:
            st.warning(f"📩 {st.session_state.mveqqeti_qeydiyyat_data['email']} ünvanına 6 rəqəmli kod göndərildi.")
            with st.form("otp_yoxlama_formu"):
                daxil_edilen_otp = st.text_input("Kodu daxil edin", type="password")
                onayla_btn = st.form_submit_button("Qeydiyyatı Tamamla və Giriş Et ✅")
                
                if onayla_btn:
                    if daxil_edilen_otp == st.session_state.otp_kodu:
                        data = st.session_state.mveqqeti_qeydiyyat_data
                        with psycopg2.connect(st.secrets["DB_URL"]) as conn:
                            with conn.cursor() as cursor:
                                cursor.execute(
                                    "INSERT INTO istifadeciler (ad_soyad, email, sifre, cihaz_hash) VALUES (%s, %s, %s, %s) RETURNING id",
                                    (data["ad_soyad"], data["email"], data["sifre"], cihaz_iz)
                                )
                                yeni_user_id = cursor.fetchone()[0]
                                conn.commit()
                        
                        st.query_params["faberlic_user_name"] = data["ad_soyad"]
                        st.query_params["faberlic_user_uuid"] = str(yeni_user_id)
                        
                        st.session_state.otp_gonderildi = False
                        st.session_state.otp_kodu = None
                        
                        st.success("🎉 Qeydiyyatınız uğurla tamamlandı!")
                        st.rerun()
                    else:
                        st.error("❌ Yanlış OTP kod daxil etdiniz.")
                        
            if st.button("⬅️ Geri qayıt və məlumatları düzəlt"):
                st.session_state.otp_gonderildi = False
                st.rerun()

else:
    st.sidebar.title(f"👤 {user_full_name}")
    st.sidebar.write(f"🆔 ID: {user_uuid}")
    
    if st.sidebar.button("🔴 Məni Unut (Çıxış)"):
        st.query_params.clear()  
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
                    img_desc = st.session_state.get("file_infos")
                else:
                    img_desc = None
                bot_response = bot.ask(question=user_input, image_description=img_desc)
                
                if "last_generated_video" in st.session_state:
                    st.video(st.session_state.last_generated_video)
                    st.session_state.messages.append({"role": "assistant", "content": st.session_state.last_generated_video, "type": "video"})
                
                st.markdown(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response, "type": "text"})