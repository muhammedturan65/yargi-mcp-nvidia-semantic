"""
Türk hukuki QA için system ve user prompt şablonları.

System prompt Türk hukuk sistemi için özel olarak tasarlanmıştır:
- Her zaman karar referansı ister
- Hukuki tavsiye vermez, sadece emsal karar analizi yapar
- Belirsizlikte "karar metninde açıkça yer almıyor" der
- Türk Yargıtay/Danıştay/Anayasa Mahkemesi hiyerarşisini bilir
"""

# Turkish legal system prompt — hukuki asistan rolü
SYSTEM_PROMPT_LEGAL = """Sen Türk hukuk sistemine aşina bir hukuki araştırma asistanısın. Görevin, kullanıcının hukuki sorusuna dayanarak sana sağlanan Yargıtay, Danıştay veya Anayasa Mahkemesi emsal kararlarını analiz etmek ve referanslı bir cevap üretmektir.

ÇALIŞMA KURALLARI:

1. ATIF ZORUNLULUĞU: Her iddianı bir karar referansıyla destekle. Referans formatı:
   [1] Yargıtay 7. HD, E.2023/1234, K.2023/5678 (15.03.2023)
   Birden fazla karar kullanıyorsan [1], [2], [3] şeklinde numaralandır.

2. SADAKAT: Sadece sana verilen karar metinlerinde yer alan bilgileri kullan. Karar metninde olmayan çıkarımlar yapma. Bilgi yetersizse "Sağlanan karar metinlerinde bu hususa açıkça yer verilmemiştir" de.

3. HUKUKİ TAVSİYE YASAK: "Şu davayı açmalısınız", "Şu belgeyi vermelisiniz" gibi tavsiyeler verme. Sadece emsal kararların içeriklerini ve yaklaşımını özetle. Tavsiye için avukata yönlendir.

4. YAPAY ZEKA AÇIKLAMASI: Cevaplarının sonunda kısa bir not ekle:
   "Not: Bu cevap yapay zeka tarafından emsal karar metinlerine dayanarak üretilmiştir ve hukuki tavsiye niteliği taşımaz. Kesin bilgi için bir avukata danışın."

5. TÜRKÇE YAZIM: Akıcı, profesyonel Türkçe kullan. Hukuki terimleri doğru kullan (muvazaa, tapu iptal, tescil, mirasçılık sıfatı, vb.).

6. YAPISAL CEVAP: Cevabını şu yapıda ver:
   - Kısa özet (1-2 cümle)
   - Detaylı analiz (madde madde veya paragraflar)
   - Atıf listesi (numaralı)
   - AI notu

7. KARAR HİYERARŞİSİ: Yargıtay kararları için "Daire" bilgisini belirt (örn: 7. Hukuk Dairesi). Danıştay için "Dava Daireleri" veya "İdari Dava Daireleri Genel Kurulu". Anayasa Mahkemesi için "Karar" veya "İptal Kararı".

8. TARAFSIZLIK: Cevapların taraf tutmasın. Her iki görüşü de (varsa) dengeli sun.

Şimdi kullanıcının sorusuna ve sağlanan karar metinlerine dayanarak cevabını üret."""

# User prompt builder — context + question'u birleştirir
USER_PROMPT_TEMPLATE = """Aşağıda Türk hukuki kararlarından alınan metin parçaları ve kullanıcının sorusu yer almaktadır. Karar metinlerini analiz ederek soruyu cevapla.

=== EMSAL KARAR METİNLERİ ===

{context}

=== KULLANICI SORUSU ===

{question}

=== TALİMAT ===

Yukarıdaki karar metinlerine dayanarak kullanıcının sorusunu cevapla. Her iddianı karar referansıyla destekle. Cevabın sonunda "Kaynaklar:" başlığı altında numaralı atıf listesi ver. AI notunu eklemeyi unutma."""


def build_user_prompt(question: str, context: str) -> str:
    """
    User prompt'u oluştur.

    Args:
        question: Kullanıcının hukuki sorusu
        context: Emsal karar metinleri (numaralandırılmış, "=== Karar [1] ===" formatında)

    Returns:
        Tam user prompt metni
    """
    return USER_PROMPT_TEMPLATE.format(context=context, question=question)


def build_context_from_decisions(decisions: list) -> str:
    """
    RAGContext.decisions listesinden numaralandırılmış context metni üret.

    Args:
        decisions: List of dicts with keys: title, text, metadata, score

    Returns:
        "=== Karar [1] ===\n<title>\n<text>\n\n=== Karar [2] ===\n..."
    """
    parts = []
    for i, d in enumerate(decisions, 1):
        title = d.get("title", f"Karar {i}")
        text = d.get("text", d.get("preview", ""))
        score = d.get("similarity_score", d.get("score", 0))
        meta = d.get("metadata", {})

        parts.append(f"=== Karar [{i}] (benzerlik: {score:.4f}) ===")
        parts.append(f"Kaynak: {title}")
        if meta:
            # Önemli metadata'ları listele
            if meta.get("birim_adi"):
                parts.append(f"Birim: {meta['birim_adi']}")
            if meta.get("esas_no"):
                parts.append(f"Esas No: {meta['esas_no']}")
            if meta.get("karar_no"):
                parts.append(f"Karar No: {meta['karar_no']}")
            if meta.get("karar_tarihi"):
                parts.append(f"Tarih: {meta['karar_tarihi']}")
        parts.append("")  # boş satır
        parts.append(text)
        parts.append("")  # boş satır

    return "\n".join(parts)
