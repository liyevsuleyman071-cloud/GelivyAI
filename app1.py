import psycopg2
import requests
from fastapi import FastAPI, Request, Response
from app import Gelivy

app = FastAPI()

def get_db_connection():
    return psycopg2.connect("MÖVCUD_POSTGRESQL_BAZA_LINKINIZ")

@app.get("/webhook")
def facebook_verify(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    
    if mode == "subscribe" and token == "gelivy_gizli_sifre":
        return Response(content=challenge, media_type="text/plain")
    return Response(content="Doğrulama uğursuz oldu", status_code=403)

@app.post("/webhook")
async def instagram_comment_webhook(request: Request):
    data = await request.json()
    
    try:
        if "entry" in data and "changes" in data["entry"][0]:
            entry = data["entry"][0]
            instagram_account_id = entry["id"]
            
            change = entry["changes"][0]
            if change["field"] == "comments":
                comment_value = change["value"]
                comment_id = comment_value["id"]      
                gelen_sual = comment_value["text"]    
                media_id = comment_value["media"]["id"] 
                
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT insta_page_access_token, insta_aktiv_mi 
                    FROM istifadeciler 
                    WHERE insta_business_account_id = %s
                """, (instagram_account_id,))
                user_data = cursor.fetchone()
                cursor.close()
                conn.close()
                
                if not user_data or not user_data[1]:
                    return Response(content="User Not Active", status_code=200)
                
                user_token = user_data[0] 
                
                video_url = f"https://graph.facebook.com/v19.0/{media_id}"
                video_res = requests.get(video_url, params={"fields": "caption", "access_token": user_token}).json()
                video_aciqlamasi = video_res.get("caption", "")
                Gelivy 
                bot=Gelivy(comment_id,prompt=f"""Sən "Gelivy AI" tərəfindən quraşdırılmış professional və köməkçil bir Instagram satış asistentisən. 
Sənə bir müştərinin sualı və həmin şərhin yazıldığı videonun altındakı rəsmi açıqlama (caption) mətni veriləcək.

Səndən tələb olunan qaydalar:
1. Müştərinin sualına yalnız və yalnız videonun altındakı rəsmi açıqlama mətninə əsasən cavab ver. 
2. Əgər videonun altında qiymət, rəng, çatdırılma və ya ölçü barədə məlumat varsa, sualı dərhal və dəqiq cavablandır.
3. Əgər müştərinin soruşduğu məlumat (məsələn, qiymət) video mətnində YOXDURSA, uydurma rəqəm demə!
4. Cavabların qısa, aydın, səmimi və maksimum 2-3 cümlədən ibarət olsun.
5. Uyğun yerlərdə pozitiv emojilərdən (✨, 🌸, 🛍️, ✅) istifadə et.
6. Müştəri hansı dildə yazırsa (Azərbaycanca, Rusca və ya Türkcə), ona həmin dildə cavab ver.

---
VİDEO MƏTNİ:
{video_aciqlamasi}

İndi bu qaydalara uyğun olaraq müştəriyə yazılacaq son cavab mətnini generasiya et (əlavə heç bir giriş və ya izah yazma, sadəcə cavabın özünü ver):""")
                bot_cavabi = bot.ask(gelen_sual)
                bot_cavabi = "Salam! Sualınız üçün təşəkkürlər. Maraqlandığınız məhsul barədə məlumat Direct qutunuza göndərildi! ✨" 
                
                reply_url = f"https://graph.facebook.com/v19.0/{comment_id}/replies"
                requests.post(reply_url, params={"access_token": user_token}, json={"message": bot_cavabi})
                
    except Exception as e:
        print(f"Sistem xətası baş verdi: {e}")
        
    return Response(content="EVENT_RECEIVED", status_code=200)