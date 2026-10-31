# pusula: NihaiAlgoritma'nın yeniden kuruluşu (inceleme ve gerekçe)

> 2026-10-23. Bu belge projenin neden ve nasıl yeniden kurulduğunu anlatır. Kod, README ve
> teknik belgeler İngilizce; bu belge Türkçe.

## 1. İstek ve kapsam

İstek: NihaiAlgoritma'yı Jane Street / Citadel Securities düzeyine çıkaracak şekilde **baştan**
geliştirmek; **fikir aynı kalacak** (market maker olmayacak), **yeni bir proje** olacak.

Bir gün önceki deneme (`Desktop\jane`, paket `meridian`) projeyi bir piyasa yapıcı
laboratuvarına çevirmişti (L3 emir defteri, Avellaneda–Stoikov). Bu, isteğin "fikir aynı kalsın"
kısmıyla çelişiyordu. `pusula` bu yüzden ayrı bir proje: NihaiAlgoritma'nın kendi fikrini, yani
**yönlü hisse (ve opsiyon) sinyal/al-sat sistemini** koruyor ve onu kurumsal çıtada yeniden kuruyor.

NihaiAlgoritma'ya hiçbir şey yazılmadı: ana süreç o çalışma ağacından koşuyor. Oradan yalnız
salt okunur iki şey alındı: Alpaca anahtarlarının bulunduğu `.env` (yalnız çalışma anında, ortam
değişkeni olarak okunur; hiçbir dosyaya kopyalanmadı) ve DoltHub IV geçmişi önbelleği (kaynak
yolu ve SHA-256'sı kaydedilerek pusula'nın kendi deposuna aktarılır).

## 2. Korunan fikir

| NihaiAlgoritma | pusula karşılığı |
|---|---|
| Evreni tara, olay/trend sinyali üret | PIT S&P 500 evreni (2016'dan beri 731 güvenlik), nedensel sinyal kütüphanesi |
| Canlı strateji `quant_trend` (21/63/252 trend işareti + EWMA σ boyutlandırma) | `TimeSeriesMomentum` + `TrendFollowing` (ters oynaklık boyutlandırma, portföy oynaklık hedefi); ailede `trend_heritage` olarak ön-kayıtlı |
| Risk yöneticisi (Madde 2, 9, 10, 12-14, 20, 42) | ön-işlem risk kapısı (Reg T brüt ≤ 2×, isim/sektör/net/beta tavanları, $5 fiyat tabanı, kısıtlı liste, ödünç zor isimlerde açığa satış artışı yasağı, ADV katılım tavanı, devir tavanı) + kilitlenen kill-switch |
| Opsiyon modu (sinyal → delta ~0,5 call, 30-45 DTE) | `options/`: BSM fiyat/Greek, IV çözücü, listelenen kullanım fiyatı ızgarası, **gerçek IV** ile fiyatlanan call katmanı; ayrı ön-kayıtlı aile `us-options-v1` |
| AI katmanı (ML, sinir ağları) | yürüyen-pencere, purge + embargo'lu Ridge ve gradyan artırmalı ağaç; doğrusal birleşimi aynı kapıdan geçerek yenmek zorunda |
| Gölge kasalar (mekanizma başına 1M$ kâğıt kasa) | her ön-kayıtlı deneme için kayıttan sonraki ilk seansta açılan **ileri kâğıt kasa** (geriye doldurulamaz) |
| Pano + Telegram | salt-okunur web panosu (127.0.0.1) |
| Ön-kayıt, DSR, exec-lag kapısı (elle, CLAUDE.md'de) | **kodda**: hash zincirli ön-kayıt defteri, otomatik deneme sayımı, gecikme ve maliyet stresi, PBO |

## 3. "Jane Street / Citadel düzeyi" burada neye karşılık geliyor

Her iki kurumun sitesi incelendi. İkisi de çekirdekte piyasa yapıcı; buradan alınan şey iş
modelleri değil, **mühendislik ve araştırma çıtası**: kurum içi tahmin modelleri, ölçeklenen ve
karmaşıklığı yönetilen sistem tasarımı, kurum içi risk sistemleri, tipli ve test edilmiş kod,
araştırma–teknoloji iç içeliği. Bu proje bu çıtayı şu somut özelliklerle karşılıyor:

1. **Nokta-zamanlı veri.** Survivorship'ten arınmış evren; çıkarılan üyeler, son endeks günündeki
   varlık olarak (`asof`) fiyatlanıyor; değişmeyen güvenlik kimlikleri; veri manifesti + SHA-256.
2. **Kalibre edilmiş maliyetler.** Makas modeli 439 gerçek NBBO örneğiyle kalibre edildi; tek
   parametreli literatür modelinin 2022 sonrası makasları 2,2 kat eksik tahmin ettiği **ölçüldü**
   ve reddedildi (docs/CALIBRATION.md).
3. **Doğrulanmış risk modeli.** İstatistiksel faktör modeli + oynaklık rejimi ayarı; tahmin
   yanlılığı gerçek veride ölçüldü (1,13 → 0,98); kalan %8'lik uzun-kısa eksik tahmini
   belgelendi.
4. **Maliyet farkındalıklı optimizasyon.** Faktör formunda risk, doğrusal makas + |Δw|^1,5
   etki maliyeti, katılım ve uygunluk garantisi; DPP ile bir kez derlenip yeniden kullanılıyor.
5. **Tek kod yolu.** Araştırma motoru ile üretim (kâğıt) motoru aynı strateji nesnesini çağırıyor;
   parite testi, tam tamsayı defter, hash zincirli günlük, yeniden oynatma.
6. **Mekanik dürüstlük.** Look-ahead denetleyicisi her sinyale uygulanıyor; ön-kayıt defteri,
   DSR, PBO, gecikme/maliyet stresi, FF5+MOM alfa testi kodda.
7. **Kalite kapısı.** `ruff` + `mypy --strict` + 100'ü aşkın test (property-based, Monte Carlo
   kalibrasyonlu istatistik testleri, mimari katman kuralı, determinizm).

Bir kişinin bir oturumda bu kurumların *ölçeğine* (binlerce kişi, borsa bağlantıları, tick
verisi) ulaşamayacağı açık; ulaşılabilecek olan **yöntem ve disiplin çıtasıdır**, bu proje onu
hedefliyor ve nerede kaldığını sayılarla söylüyor.

## 4. Geliştirme sırasında bulunan ve düzeltilen gerçek hatalar

Her biri bir testin ya da bir ölçümün yakaladığı, sessizce yanlış sonuç üretecek hatalardı:

| # | bulgu | etki | düzeltme |
|---|---|---|---|
| 1 | Alpaca çoklu sembol isteği DOW'u (DD ile aynı istekte) **sessizce düşürüyor** | bir güvenliğin tüm geçmişi kayıp | boş dönen her sembol tek başına yeniden istenir |
| 2 | Alpaca 22 güvenlikte 5.899 barda `vwap = 0` | önce tüm bar siliniyordu | alan düzeyinde onarım (yalnız VWAP NaN) |
| 3 | Onarım `in_universe` maskesine de dokunuyordu | üyelik maskesi NaN | yalnız bar alanları onarılır |
| 4 | Kotasyon tablosu yeni indeksle kurulunca pandas hizalaması tüm değerleri NaN yapıyordu | kalibrasyon 0/40 | `.to_numpy()` |
| 5 | Dolum, onaylanmamış emirde reddedilirken miktar **önce** artırılıyordu | emir defteri bozulması | geçiş önce doğrulanır |
| 6 | Clarabel warm-start: çözüm geçmişe bağımlı | yeniden başlatılan kasa kesintisiz koşudan ~1.000 olay sonra ayrışıyordu | `warm_start=False` + determinizm testi |
| 7 | Endeksten çıkan hissenin kimliği değişiyordu | ileri kasada açık pozisyon kimliksiz kalırdı | kimlik = `TICKER@başlangıç` |
| 8 | OneDrive senkronu atomik yeniden adlandırmayı kilitliyordu | veri deposu yazılamıyordu | veri `%LOCALAPPDATA%\pusula`'da + yeniden denemeli atomik yazma |
| 9 | Wikipedia httpx başlıklarını 403 ile reddediyor | evren kurulamıyordu | stdlib istemcisi |
| 10 | 8 işçinin her biri tüm çekirdeklerde BLAS koşturuyordu (aşırı abonelik) | 10 ms'lik SVD 750 ms; çalışma saatlerce sürecekti | işçide `OMP/OPENBLAS/MKL_NUM_THREADS=1` |
| 11 | Uzun süren çalışma, başlarken okuduğu defter başına zincirliyordu; araya başka kayıt girince zincir çatallanacaktı | defter bütünlüğü | çapraz süreç kilidi + eklemeden önce diskten yeniden okuma; iki yazıcı testi |
| 12 | **Sharpe/t/PSR/DSR/PBO ham getiriden hesaplanıyordu** (fazla getiri yerine) | `us-equity-v1`'de kill-switch ile nakde geçen üç kitap T-bill kazanırken Sharpe 12–83 göründü; DSR/PBO kirlendi (kapı yine de terfi ettirmedi) | tüm istatistikler fazla getiriden; sabit seride Sharpe tanımsız (NaN); regresyon testi; v1 aynı yapılandırmayla gerekçeli yeniden koşuldu |
| 13 | Araştırma backtest'inde **kilitlenen** kill-switch | tetiklendikten sonraki örneklem stratejiyi değil nakdi ölçüyor | v1 değiştirilmedi; kill-switch'siz `us-equity-v2` ayrıca kaydedildi, DSR v1+v2 (18 deneme) üzerinden; ileri kasalarda kill-switch açık |
| 14 | 2× maliyet stresinde optimizöre de maliyetlerin arttığı söyleniyordu | optimizör daha az işlem yapıp stresi boşa çıkarıyordu (stresli t temel t'den yüksek çıktı) | strateji kayıtlı maliyet modeliyle karar verir, motor 2× tahsil eder; test |
| 15 | DSR: deneme sayısı programdan, Sharpe varyansı yalnız aileden | v3'ün 3 benzer denemesi DSR 0,86 aldı (gerçekte 0,36) | varyans da program düzeyinde; `study reevaluate` ile defterden yeniden hesaplandı |
| 16 | Koşu, koddaki denemenin kayıtlı yapılandırmayla aynı olduğunu doğrulamıyordu | kayıttan sonra değiştirilmiş bir deneme eski kaydın altında koşabilirdi | `verify_registration`: hash farklıysa koşu reddedilir |
| 17 | Kaldığı yerden devam eden ileri kasanın özsermaye eğrisi yalnız o süreçte işlenen günleri içeriyordu | üretimde her gece yeni süreç olduğundan panodaki eğri ve "açılıştan beri" getiri her gece sıfırlanırdı | eğri devamda günlükteki fiyatlama olaylarından (tam tamsayı) yeniden kuruluyor; iki gecelik entegrasyon testi |
| 18 | Risk kapısında sektör net onarımı tüm isimleri eşit kaydırıyordu | ağırlık sınırındaki bir kısa pozisyon %5'ten %6,67'ye çıkabiliyordu (kayıtlı ailelerde sektör limiti kullanılmadığı için sonuçlara etkisi yok) | aşıma neden olan taraf orantılı küçültülüyor; regresyon testi |
| 19 | Konteyner düzeninde `docs/` yok; rapor komutu dosya bulunamadı hatasıyla çöküyordu (Dockerfile adımları temiz klasörde yeniden oynatılarak bulundu) | `study report/run` Docker'da çalışmazdı | klasör gerektiğinde oluşturuluyor; defter ve raporlar kalıcı birimlerde |
| 20 | IV deposu olmayan ortamda (konteyner) opsiyon-sinyal ailesi gece koşusunu çökertirdi | tüm ileri kasalar o gece işlenmezdi | IV gerektiren aile açık mesajla atlanıyor; `data iv-update` depoyu DoltHub toplu CSV'sinden sıfırdan kurabiliyor (gerçek veride: 1.842.608 satır, artımlı depoyla birebir aynı) |
| 21 | Veri deposu `%LOCALAPPDATA%` altındaydı; MSIX paketli bir uygulamadan (Claude masaüstü) başlatılan süreçlerin `AppData` yazmaları paketin özel kopyasına (`Packages\…\LocalCache`) yönlendiriliyor, Görev Zamanlayıcı ise gerçek klasörü görüyor (ilk zamanlanmış koşu `universe_meta.json` bulunamadı diye çöktü) | gece görevi hiçbir kasayı ilerletemezdi | varsayılan `~/.pusula` (ne OneDrive ne `AppData`); depo oraya kopyalandı, görev yeniden koşturularak doğrulandı |
| 22 | Endeksten çıkarılan üyelerin CIK eşlemesinde ad benzerliği KISA ada bölünüyordu: "Coca-Cola Enterprises" → Coca-Cola Co (KO), "Time Warner" ve "Time Warner Cable" → Time Inc., "People's United Financial" → Peoples Financial /MS/, "First Republic Bank" → Republic First Bancorp (FRC FDIC'e dosyalar, EDGAR kaydı YOK) | yanlış şirketin bilançosu sinyale girerdi | benzerlik endeks adına göre (endeks adının her kelimesi EDGAR adında, şimdiki ya da eski) + ilk kelime eşit; regresyon testi |
| 23 | İlk doğrulama "spell'e yakın HERHANGİ bir rapor"u yeterli sayıyordu: AIV (2016) → 2020'de kurulan Aimco OP L.P., BHI (2016) → 2017'de kurulan Baker Hughes Co, DD (2016) → DowDuPont, OI → borç ihraç eden yan şirket | halef/yan kuruluşun verisi seleflerin dönemine yazılırdı | aday, spell'in HEM başına HEM sonuna 400 gün içinde 10-K/10-Q vermiş olmalı; eşitlikte hâlâ o kodu taşıyan kuruluş seçilir; 28 elle doğrulanmış istisna gerekçeleriyle kodda, onlar da aynı denetimden geçiyor |
| 24 | pandas 3'te `astype(str)` eksik değeri "nan" metnine çevirmiyor, NaN bırakıyor: CIK'siz satırlı ana tablonun özeti NaN'ı hash'leyemedi | depo özeti kurulamazdı (sessiz değil, hata verdi) | satırlar açıkça kuruluyor |
| 25 | Gece koşusunda SEC'e ulaşılamazsa güncelleme istisnası BÜTÜN kasaları durdururdu | fiyat tabanlı kasalar da o gece işlenmezdi | yalnız temel-veri ailesi o gece atlanır (eski depoyla işlem görmez), diğerleri ilerler, çıkış kodu 1; olaylar kabul zamanını taşıdığı için kaçan seanslar sonraki iyi gecede doğru bilgiyle oynatılır |
| 26 | Risk modeline varsayılanı kapalı üç alan (`eigen_adjust`, `eigen_sims`, `industry_min_size`) eklenince kayıtlı her denemenin yapılandırma özeti değişti: kayıtlı yapılandırma, iç içe `RiskModelSpec`i TAM yazıyordu (yalnız Trial alanları varsayılanda atlanıyordu) | v1-v3, opsiyon-sinyal ve temel-veri aileleri koşmayı reddeder, gece koşusu düşerdi (canlıya çıkmadan yakalandı) | sonradan eklenen risk alanları da varsayılandayken atlanıyor; kodun her ailesini depodaki deftere karşı doğrulayan test eklendi (bu kusuru o yakalardı) |
| 27 | Yayımlanan risk modeli kalibrasyonu (CALIBRATION §2) 10 seansta bir tek bir başlangıç satırından ölçülmüştü: eşit ağırlıkta yanlılık başlangıç satırına göre 0,86-1,17 arasında oynuyor | "uzun-kısa portföylerde %8 eksik tahmin" büyük ölçüde gürültüydü (her seansla %3); asıl sorun olan optimizasyon yanlılığı (%21) hiç ölçülmemişti | her seansla yeniden ölçüldü ve düzeltme tabloyla yayımlandı; kalibrasyon artık betikle (`tools/risk_bias.py`), optimize portföyler dahil |
| 28 | FINRA açığa satış dosyaları hisseyi O GÜNKÜ koduyla adlandırıyor, menkul kıymet kimliği ise bugünkü kodu taşıyor. İlk depo yalnız bugünkü kodları tutuyordu: yeniden adlandırılan 26 üye adlandırmadan önce veri almıyordu (FB→META, BK→BNY, FLT→CPAY…), 7'sinin bugünkü kodu ise önce BAŞKA bir şirketindi (BNY bir belediye tahvili fonu, GEN Genesis Healthcare, DOC Physicians Realty, COR CoreSite, CTRA Contura, ECHO Echo Global Logistics, META 2021'de bir metaverse ETF'i) | başka şirketin açığa satış hacmi o üyenin sinyaline girerdi (koşudan ve kayıttan ÖNCE yakalandı) | her üye yılı kendi barlarıyla günlük hacim değişimi korelasyonuyla denetleniyor (aynı şirket ~0,72, yabancı <0,33); depo artık tüm sembolleri tutuyor, eski kodlar veriden bulunup tarihleriyle `ticker_history.py`'de; satır yalnız menkulün fiyatı olan seansta sayılıyor; çeyrek düzeyinde 23.910 sembolün tamamına karşı arama başka eşleşme bulmadı (`docs/SHORTVOL_DATA.md`) |
| 29 | Gece koşusunda DoltHub IV güncellemesi istisna yakalamıyordu (SEC/FINRA'nınki yakalıyordu): 2026-10-27'de DoltHub SQL API'si "context deadline exceeded" verdi | o gece HİÇBİR kasa ilerlemedi, FINRA deposu da güncellenmedi (görev sonucu 1; makine uykudan uyanınca kendiliğinden başlayan telafi koşusunda görüldü) | IV hatası yalnız opsiyon-ima ailesini o gece atlatır, diğerleri ilerler, çıkış 1; üç kaynak (SEC, FINRA, DoltHub) için aynı test parametreli. Kök neden ölçüldü: baş commit'i `dolt_log ORDER BY date DESC LIMIT 1` ile soruyordu (tüm commit geçmişini sıralar, sunucu sınırı ~56 sn'de düştü, yeniden üretildi); artık `dolt_branches` (0,7 sn, aynı hash) ve yalnız yeni satır varsa |

## 5. Bilinen sınırlar (sonuçlardan önce yazıldı)

* Veri günlük bar (Alpaca SIP, 2016'dan). Gün içi kuyruk/kotasyon simülasyonu yok; infaz
  VWAP + kalibre makas + karekök etki ile modelleniyor. Etki katsayısı kalibre değil → kapı
  2× maliyet stresi istiyor.
* Spin-off'lar satıcı tarafından düzeltilmiyor (AA 2016, TGNA 2017 gibi), kalite raporunda
  bayraklanıyor.
* GICS sektörü yalnız güncel üyeler için biliniyor; risk modeli istatistiksel (PCA) olduğu için
  sektör bilgisine dayanmıyor.
* Opsiyonlar tek ATM IV ile fiyatlanıyor (gülümseme yok), makas primin sabit bir oranı,
  temettü sabit getiri varsayımıyla.
* French faktör verisi 2026-08-31'de bitiyor; ayrılmış dönemin son ~7 haftası faktör
  regresyonuna giremiyor.
