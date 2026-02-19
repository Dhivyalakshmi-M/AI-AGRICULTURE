import streamlit as st
from difflib import get_close_matches

def main():

    # Agricultural knowledge base with 30 Q&A pairs in English and Tamil
    agriculture_kb = {
        'English': {
            'questions': [
                "What are the government schemes for farmers?",
                "How to get PM Kisan benefits?",
                "What is the crop insurance scheme?",
                "How to check weather forecast for farming?",
                "When is the best time to sow paddy?",
                "What fertilizer for sugarcane?",
                "How to control stem borer in paddy?",
                "What are organic farming methods?",
                "How to increase soil fertility?",
                "What is the market price for wheat today?",
                "How to prevent fungal diseases in crops?",
                "What is drip irrigation?",
                "How to apply for agricultural loan?",
                "What are the best crops for black soil?",
                "How to manage water scarcity?",
                "What pesticides for cotton crops?",
                "How to control weeds naturally?",
                "What is crop rotation?",
                "How to store grains properly?",
                "What are the symptoms of nitrogen deficiency?",
                "How to grow tomatoes organically?",
                "What is minimum support price?",
                "How to treat leaf curl disease?",
                "What are the latest farming technologies?",
                "How to prepare vermicompost?",
                "What is precision farming?",
                "How to control rats in farm?",
                "What are the benefits of mulching?",
                "How to test soil pH?",
                "What are the subsidies for farm equipment?"
            ],
            'answers': [
                "Government schemes include PM-KISAN, PMFBY, KCC, National Mission on Sustainable Agriculture, Paramparagat Krishi Vikas Yojana, and Soil Health Card Scheme.",
                "Register on PM-KISAN portal with Aadhaar, land records, and bank details. Visit local agriculture office or CSC center for assistance.",
                "PMFBY provides comprehensive insurance cover for crops against natural calamities. Premium is 2% for Kharif, 1.5% for Rabi, and 5% for commercial crops.",
                "Use IMD website, mobile apps like Meghdoot, or SMS service by sending SMS to 51969 with district code. Also check Krishi Vigyan Kendra updates.",
                "For Kharif paddy: June-July with onset of monsoon. For Rabi paddy: November-December in areas with irrigation facilities.",
                "Apply 150:60:60 kg NPK per hectare. Use 25% N as basal dose, remaining in splits. Include zinc sulfate 25 kg/ha and iron sulfate 50 kg/ha.",
                "Use neem-based pesticides, install pheromone traps, release Trichogramma wasps, apply carbofuran 3G, and practice field sanitation.",
                "Methods include crop rotation, green manure, vermicomposting, biological pest control, and using organic fertilizers like Jeevamrutham.",
                "Add organic matter, practice green manuring, use compost, maintain proper pH, implement crop rotation, and avoid chemical overuse.",
                "Current wheat MSP is Rs 2275 per quintal for 2024-25. Market prices vary by quality and location, check eNAM portal for real-time rates.",
                "Use resistant varieties, ensure proper spacing, avoid water logging, apply Trichoderma, use copper-based fungicides, and practice crop rotation.",
                "Drip irrigation delivers water directly to plant roots through pipes with emitters. It saves 30-50% water and increases yield by 20-30%.",
                "Approach banks with land documents, ID proof, project report, and KCC. NABARD offers 3% interest subvention for timely repayment.",
                "Cotton, soybean, sugarcane, wheat, and pulses grow well in black soil (regur soil) due to high clay content and moisture retention.",
                "Use drip/sprinkler irrigation, grow drought-resistant crops, practice mulching, harvest rainwater, and implement soil moisture conservation.",
                "For sucking pests: Imidacloprid 17.8% SL. For bollworms: Emamectin benzoate 5% SG. Always follow recommended dosage and safety period.",
                "Use manual weeding, mulching with organic materials, allelopathic crops like sorghum, and natural herbicides from vinegar or corn gluten.",
                "Growing different crops sequentially on same land to improve soil health, break pest cycles, and increase biodiversity. Example: Rice-Wheat-Pulses.",
                "Clean and dry grains to 12-14% moisture, use airtight containers, add neem leaves, store in cool dry place, and fumigate if needed.",
                "Yellowing of older leaves, stunted growth, reduced tillering, poor flowering. Correct with urea application or organic nitrogen sources.",
                "Use organic seeds, prepare raised beds, apply vermicompost, use neem oil for pests, practice companion planting with marigold.",
                "MSP is guaranteed price by government for 23 crops to protect farmers from price fluctuations. Announced before sowing seasons.",
                "Remove infected leaves, spray neem oil solution, apply systemic fungicides, maintain proper spacing, and ensure good air circulation.",
                "Drone spraying, IoT sensors, AI-based disease detection, automated irrigation, vertical farming, and hydroponics are emerging technologies.",
                "Layer organic waste with earthworms in shaded pits. Maintain moisture and temperature. Harvest after 45-60 days. Ratio: 1 kg worms for 100 kg waste.",
                "Using technology like GPS, sensors, drones to optimize inputs, monitor crops precisely, reduce waste, and increase productivity.",
                "Use traps, encourage natural predators like owls, maintain cleanliness, use rodenticides carefully, and seal storage areas.",
                "Conserves soil moisture, controls weeds, regulates temperature, improves soil structure, and adds organic matter as it decomposes.",
                "Use soil testing kit from agriculture department. Collect samples from 10-15 spots at 15cm depth. Send to lab or use digital pH meter.",
                "50% subsidy on drip irrigation, 35-50% on tractors, 40% on harvesters, 50% on solar pumps under various state and central schemes."
            ]
        },
        'Tamil': {
            'questions': [
                "விவசாயிகளுக்கான அரசு திட்டங்கள் என்ன?",
                "பி.எம். கிசான் நலன்களை எவ்வாறு பெறுவது?",
                "பயிர் காப்பீட்டு திட்டம் என்ன?",
                "விவசாயத்திற்கான வானிலை முன்னறிவிப்பை எவ்வாறு சரிபார்க்கலாம்?",
                "நெல் விதைக்க சிறந்த நேரம் எப்போது?",
                "கரும்புக்கு எந்த உரம்?",
                "நெல்லில் தண்டு துளைப்பான் கட்டுப்படுத்த எப்படி?",
                "கரிம விவசாய முறைகள் என்ன?",
                "மண் வளத்தை எவ்வாறு அதிகரிப்பது?",
                "இன்று கோதுமைக்கான சந்தை விலை என்ன?",
                "பயிர்களில் பூஞ்சை நோய்களை எவ்வாறு தடுப்பது?",
                "டிரிப் பாசனம் என்றால் என்ன?",
                "விவசாய கடனுக்கு விண்ணப்பிக்க எப்படி?",
                "கருமண்ணுக்கு சிறந்த பயிர்கள் எவை?",
                "நீர் பற்றாக்குறையை எவ்வாறு நிர்வகிப்பது?",
                "பருத்திப் பயிர்களுக்கு என்ன பூச்சிக்கொல்லிகள்?",
                "களைகளை இயற்கையாக கட்டுப்படுத்த எப்படி?",
                "பயிர் சுழற்சி என்றால் என்ன?",
                "தானியங்களை சரியாக எவ்வாறு சேமிப்பது?",
                "நைட்ரஜன் குறைபாட்டின் அறிகுறிகள் என்ன?",
                "தக்காளியை கரிம முறையில் எவ்வாறு வளர்ப்பது?",
                "குறைந்தபட்ச ஆதரவு விலை என்றால் என்ன?",
                "இலை சுருள் நோயை எவ்வாறு சிகிச்சையளிப்பது?",
                "சமீபத்திய விவசாய தொழில்நுட்பங்கள் என்ன?",
                "வெர்மிகம்போஸ்ட் தயாரிக்க எப்படி?",
                "துல்லியமான விவசாயம் என்றால் என்ன?",
                "வயலில் எலிகளை எவ்வாறு கட்டுப்படுத்துவது?",
                "மல்ச்சிங் நன்மைகள் என்ன?",
                "மண் pH ஐ எவ்வாறு சோதிப்பது?",
                "விவசாய உபகரணங்களுக்கான மானியங்கள் என்ன?"
            ],
            'answers': [
                "பி.எம்-கிசான், பி.எம்.எஃப்.பி.ஒய், கே.சி.சி, தேசிய நிலையான விவசாய திட்டம், பாரம்பரிய விவசாய மேம்பாட்டு யோஜனா, மண் ஆரோக்கிய அட்டை திட்டம் ஆகியவை அடங்கும்.",
                "ஆதார், நில பதிவேடுகள், வங்கி விவரங்களுடன் பி.எம்-கிசான் போர்ட்டில் பதிவு செய்யவும். உதவிக்கு உள்ளூர் விவசாய அலுவலகம் அல்லது சி.எஸ்.சி மையத்தைத் தொடர்பு கொள்ளவும்.",
                "பி.எம்.எஃப்.பி.ஒய் இயற்கை பேரிடர்களுக்கு எதிராக பயிர்களுக்கு விரிவான காப்பீட்டு உள்ளடக்கம் வழங்குகிறது. கரிப்பு பயிர்களுக்கு 2%, ரபி பயிர்களுக்கு 1.5%, வணிக பயிர்களுக்கு 5% பிரீமியம்.",
                "ஐ.எம்.டி வலைத்தளம், மேகதூத் போன்ற மொபைல் பயன்பாடுகள் அல்லது மாவட்ட குறியீட்டுடன் 51969 க்கு எஸ்எம்எஸ் அனுப்பும் சேவையைப் பயன்படுத்தவும். கிருஷி விக்யான் கேந்திர புதுப்பிப்புகளையும் சரிபார்க்கவும்.",
                "கரிப்பு நெல்: மழைக்கால தொடக்கத்தில் ஜூன்-ஜூலை. ரபி நெல்: பாசன வசதி உள்ள பகுதிகளில் நவம்பர்-டிசம்பர்.",
                "ஹெக்டேருக்கு 150:60:60 கிலோ NPK பயன்படுத்தவும். அடித்தளமாக 25% N பயன்படுத்தவும், மீதமுள்ளவை பிரித்து. துத்தநாக சல்பேட் 25 கிலோ/ஹெக்டேர் மற்றும் இரும்பு சல்பேட் 50 கிலோ/ஹெக்டேர் சேர்க்கவும்.",
                "வேப்ப எண்ணெய் அடிப்படையிலான பூச்சிக்கொல்லிகள் பயன்படுத்தவும், ஃபெரோமோன் பொறிகள் நிறுவவும், ட்ரைகோகிராமா குளவிகள் விடுவிக்கவும், கார்போஃபூரான் 3ஜி பயன்படுத்தவும், வயல் சுகாதாரம் பழகவும்.",
                "பயிர் சுழற்சி, பசுந்தாள், மட்டுபுழு உரம், உயிரியல் பூச்சி கட்டுப்பாடு மற்றும் ஜீவாம்ருதம் போன்ற கரிம உரங்கள் பயன்படுத்துதல் ஆகியவை அடங்கும்.",
                "கரிமப் பொருட்களைச் சேர்க்கவும், பசுந்தாள் முறையைப் பின்பற்றவும், கூட்டுஉரம் பயன்படுத்தவும், சரியான pH பராமரிக்கவும், பயிர் சுழற்சியை செயல்படுத்தவும், இரசாயன மிகைப்படுத்தல் தவிர்க்கவும்.",
                "தற்போதைய கோதுமை MSP 2024-25க்கு ஒரு குவின்டாலுக்கு ரூ 2275. சந்தை விலைகள் தரம் மற்றும் இடத்தைப் பொறுத்து மாறுபடும், நிகழ்நேர விகிதங்களுக்கு eNAM போர்ட்டைச் சரிபார்க்கவும்.",
                "எதிர்ப்பு வகைகளைப் பயன்படுத்தவும், சரியான இடைவெளி உறுதிசெய்யவும், நீர் தேங்குவதைத் தவிர்க்கவும், ட்ரைகோடெர்மா பயன்படுத்தவும், தாமிர அடிப்படையிலான பூஞ்சைக்கொல்லிகள் பயன்படுத்தவும், பயிர் சுழற்சி பழகவும்.",
                "டிரிப் பாசனம் குழாய்கள் மூலம் நேரடியாக தாவர வேர்களுக்கு நீரை வழங்குகிறது. இது 30-50% நீரைச் சேமிக்கிறது மற்றும் விளைச்சலை 20-30% அதிகரிக்கிறது.",
                "நில ஆவணங்கள், அடையாள சான்று, திட்ட அறிக்கை மற்றும் கே.சி.சி உடன் வங்கிகளை அணுகவும். நாபார்ட் சரியான நேரத்தில் திருப்பிச் செலுத்துவதற்கு 3% வட்டி மானியம் வழங்குகிறது.",
                "பருத்தி, சோயாபீன், கரும்பு, கோதுமை மற்றும் பருப்பு வகைகள் கருமண்ணில் (ரெகுர் மண்) நன்றாக வளரும், இது உயர் களிமண் உள்ளடக்கம் மற்றும் ஈரப்பதம் தக்கவைப்பு காரணமாகும்.",
                "டிரிப்/ஸ்ப்ரிங்ளர் பாசனம் பயன்படுத்தவும், வறட்சி எதிர்ப்பு பயிர்கள் வளரவும், மல்ச்சிங் பழகவும், மழைநீர் சேகரிக்கவும், மண் ஈரப்பதம் பாதுகாப்பை செயல்படுத்தவும்.",
                "உறிஞ்சும் பூச்சிகளுக்கு: இமிடாக்ளோப்பிரிட் 17.8% எஸ்எல். பால் புழுக்களுக்கு: எமாமெக்டின் பென்சோயேட் 5% எஸ்ஜி. எப்போதும் பரிந்துரைக்கப்பட்ட டோஸ் மற்றும் பாதுகாப்பு காலத்தைப் பின்பற்றவும்.",
                "கைமுறை களைகளுக்கு, கரிமப் பொருட்களுடன் மல்ச்சிங், சோளம் போன்ற அல்லேலோபாதிக் பயிர்கள் மற்றும் வினிகர் அல்லது கார்ன் குளூட்டனில் இருந்து இயற்கை பூச்சிக்கொல்லிகள் பயன்படுத்தவும்.",
                "மண்ணின் ஆரோக்கியத்தை மேம்படுத்த, பூச்சி சுழற்சிகளை முறித்து, உயிரியல் பல்வகைத்தன்மையை அதிகரிக்க ஒரே நிலத்தில் வெவ்வேறு பயிர்களை வரிசையாக வளர்ப்பது. எடுத்துக்காட்டு: நெல்-கோதுமை-பருப்பு வகைகள்.",
                "தானியங்களை 12-14% ஈரப்பதமாக சுத்தமாகவும் உலரவும் வைக்கவும், காற்றுப் புகாத கொள்கலன்களைப் பயன்படுத்தவும், வேப்பிலைகளைச் சேர்க்கவும், குளிர்ந்த உலர்ந்த இடத்தில் சேமிக்கவும், தேவைப்பட்டால் புகைபோடவும்.",
                "பழைய இலைகளின் மஞ்சள் நிறமாதல், வளர்ச்சி குன்றியது, தூற்றுதல் குறைந்தது, மலர்ச்சி குறைந்தது. யூரியா பயன்பாடு அல்லது கரிம நைட்ரஜன் மூலங்களுடன் சரிசெய்யவும்.",
                "கரிம விதைகளைப் பயன்படுத்தவும், உயர்த்தப்பட்ட படுக்கைகளைத் தயாரிக்கவும், மட்டுபுழு உரம் பயன்படுத்தவும், பூச்சிகளுக்கு வேப்ப எண்ணெய் பயன்படுத்தவும், சாமந்தி மலருடன் தோழமை விவசாயம் பழகவும்.",
                "MSP என்பது விலை ஏற்ற இறக்கங்களிலிருந்து விவசாயிகளைப் பாதுகாக்க 23 பயிர்களுக்கு அரசாங்கத்தால் உத்தரவாதம் அளிக்கப்படும் விலையாகும். விதைக்கும் பருவங்களுக்கு முன் அறிவிக்கப்படுகிறது.",
                "பாதிக்கப்பட்ட இலைகளை அகற்றவும், வேப்ப எண்ணெய் கரைசலைத் தெளிக்கவும், கணினி பூஞ்சைக்கொல்லிகளைப் பயன்படுத்தவும், சரியான இடைவெளியை பராமரிக்கவும், நல்ல காற்று சுழற்சியை உறுதிசெய்யவும்.",
                "ட்ரோன் தெளிப்பு, ஐஓடி சென்சார்கள், ஏஐ அடிப்படையிலான நோய் கண்டறிதல், தானியங்கி பாசனம், செங்குத்து விவசாயம் மற்றும் ஹைட்ரோபோனிக்ஸ் ஆகியவை எழும் தொழில்நுட்பங்கள்.",
                "நிழலான குழிகளில் மண்புழுக்களுடன் கரிமக் கழிவுகளை அடுக்குக. ஈரப்பதம் மற்றும் வெப்பநிலையை பராமரிக்கவும். 45-60 நாட்களுக்குப் பிறகு அறுவடை செய்யவும். விகிதம்: 100 கிலோ கழிவுக்கு 1 கிலோ புழுக்கள்.",
                "ஜி.பி.எஸ், சென்சார்கள், ட்ரோன்கள் போன்ற தொழில்நுட்பங்களைப் பயன்படுத்தி உள்ளீடுகளை மேம்படுத்துதல், பயிர்களை துல்லியமாக கண்காணித்தல், கழிவுகளைக் குறைத்தல் மற்றும் உற்பத்தித்திறனை அதிகரிப்பது.",
                "பொறிகளைப் பயன்படுத்தவும், ஆந்தைகள் போன்ற இயற்கை வேட்டையாடுபவர்களை ஊக்குவிக்கவும், சுத்தத்தை பராமரிக்கவும், கொறியன்கொல்லிகளை கவனமாகப் பயன்படுத்தவும், சேமிப்பு பகுதிகளை மூடவும்.",
                "மண்ணின் ஈரப்பதத்தைப் பாதுகாக்கிறது, களைகளைக் கட்டுப்படுத்துகிறது, வெப்பநிலையை கட்டுப்படுத்துகிறது, மண்ணின் அமைப்பை மேம்படுத்துகிறது, அது சிதைவடையும் போது கரிமப் பொருட்களைச் சேர்க்கிறது.",
                "விவசாயத் துறையிலிருந்து மண் சோதனை கிட்டைப் பயன்படுத்தவும். 15செமீ ஆழத்தில் 10-15 இடங்களிலிருந்து மாதிரிகளைச் சேகரிக்கவும். ஆய்வகத்திற்கு அனுப்பவும் அல்லது டிஜிட்டல் pH மீட்டரைப் பயன்படுத்தவும்.",
                "பல்வேறு மாநில மற்றும் மத்திய திட்டங்களின் கீழ் டிரிப் பாசனத்தில் 50%, டிராக்டர்களில் 35-50%, அறுவடை இயந்திரங்களில் 40%, சோலார் பம்புகளில் 50% மானியம்."
            ]
        }
    }

    def find_best_match(user_question, language):
        """Find the best matching question from knowledge base"""
        if language not in agriculture_kb:
            return None
        
        questions = agriculture_kb[language]['questions']
        if not questions:
            return None
        
        matches = get_close_matches(user_question, questions, n=1, cutoff=0.5)
        return matches[0] if matches else None

    def get_answer(user_question, language):
        """Get answer for user question"""
        if not user_question or not user_question.strip():
            if language == 'English':
                return "Please ask a question about agriculture."
            else:
                return "தயவு செய்து விவசாயம் பற்றி ஒரு கேள்வி கேளுங்கள்."
        
        best_match = find_best_match(user_question, language)
        if best_match:
            idx = agriculture_kb[language]['questions'].index(best_match)
            return agriculture_kb[language]['answers'][idx]
        else:
            if language == 'English':
                return "I can answer questions about government schemes, crop practices, weather, market prices, pesticides, or crop diseases. Please ask a specific agricultural question."
            else:
                return "நான் அரசு திட்டங்கள், பயிர் விவசாயம், வானிலை, சந்தை விலைகள், பூச்சிக்கொல்லிகள் அல்லது பயிர் நோய்கள் பற்றிய கேள்விகளுக்கு பதிலளிக்க முடியும். தயவு செய்து ஒரு குறிப்பிட்ட விவசாய கேள்வியைக் கேளுங்கள்."

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_language' not in st.session_state:
        st.session_state.current_language = 'English'
    if 'last_question' not in st.session_state:
        st.session_state.last_question = ''

    # Custom CSS for beautiful UI without emojis
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Header styling */
        .header-container {
            background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%);
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        
        .main-title {
            color: white;
            font-size: 32px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        .sub-title {
            color: #E8F5E9;
            font-size: 16px;
            font-weight: 400;
            margin-top: 0;
        }
        
        /* Language selector styling */
        .language-container {
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 25px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            border: 1px solid #C5E1A5;
        }
        
        .language-btn {
            background: #F1F8E9;
            border: 2px solid #4CAF50;
            color: #2E7D32;
            padding: 10px 25px;
            margin: 0 10px;
            border-radius: 25px;
            font-weight: 500;
            font-size: 15px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .language-btn:hover {
            background: #4CAF50;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
        }
        
        .language-btn.active {
            background: #4CAF50;
            color: white;
            font-weight: 600;
        }
        
        /* Chat container styling */
        .chat-container {
            background: white;
            padding: 0;
            border-radius: 12px;
            height: 420px;
            overflow-y: auto;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            border: 1px solid #C5E1A5;
        }
        
        .chat-inner {
            padding: 20px;
            height: 100%;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        /* Message styling */
        .message {
            max-width: 75%;
            padding: 15px;
            border-radius: 18px;
            line-height: 1.5;
            font-size: 14.5px;
            word-wrap: break-word;
        }
        
        .user-message {
            background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
            border: 1px solid #90CAF9;
            margin-left: auto;
            border-bottom-right-radius: 5px;
            color: #0D47A1;
        }
        
        .bot-message {
            background: linear-gradient(135deg, #F1F8E9 0%, #DCEDC8 100%);
            border: 1px solid #AED581;
            margin-right: auto;
            border-bottom-left-radius: 5px;
            color: #33691E;
        }
        
        .message-sender {
            font-weight: 600;
            font-size: 13px;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Quick questions styling */
        .quick-questions-container {
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            border: 1px solid #C5E1A5;
        }
        
        .section-title {
            color: #2E7D32;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #E8F5E9;
        }
        
        .question-btn {
            background: #F1F8E9;
            border: 1px solid #C5E1A5;
            color: #2E7D32;
            padding: 12px 18px;
            margin: 8px;
            border-radius: 8px;
            font-size: 14px;
            cursor: pointer;
            text-align: left;
            transition: all 0.2s ease;
            display: inline-block;
            max-width: 100%;
            word-wrap: break-word;
        }
        
        .question-btn:hover {
            background: #C8E6C9;
            border-color: #81C784;
            transform: translateY(-2px);
            box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Input area styling */
        .input-container {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            border: 1px solid #C5E1A5;
        }
        
        .input-field {
            width: 100%;
            padding: 15px;
            border: 2px solid #C5E1A5;
            border-radius: 8px;
            font-size: 15px;
            color: #333;
            transition: border-color 0.3s;
        }
        
        .input-field:focus {
            outline: none;
            border-color: #4CAF50;
        }
        
        /* Button styling */
        .action-btn {
            background: linear-gradient(135deg, #4CAF50 0%, #388E3C 100%);
            color: white;
            border: none;
            padding: 12px 28px;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 15px;
        }
        
        .action-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(76, 175, 80, 0.3);
        }
        
        .clear-btn {
            background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
            margin-left: 15px;
        }
        
        /* Knowledge section styling */
        .knowledge-container {
            background: white;
            padding: 25px;
            border-radius: 12px;
            margin-top: 25px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            border: 1px solid #C5E1A5;
        }
        
        .knowledge-column {
            padding: 15px;
            background: #F9FDF8;
            border-radius: 10px;
            height: 100%;
        }
        
        .knowledge-title {
            color: #2E7D32;
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #E8F5E9;
        }
        
        .knowledge-item {
            color: #555;
            font-size: 14px;
            padding: 6px 0;
            border-bottom: 1px dashed #E8F5E9;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            color: #666;
            font-size: 13px;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #E8F5E9;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #F1F8E9;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #C5E1A5;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #81C784;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header Section
    st.markdown("""
    <div class="header-container">
        <div class="main-title">AI Agricultural Assistant</div>
        <div class="sub-title">Expert Agricultural Knowledge in English & Tamil | Available 24/7</div>
    </div>
    """, unsafe_allow_html=True)

    # Language Selection Section
    st.markdown("""
    <div class="language-container">
        <div style="text-align: center; margin-bottom: 15px; color: #2E7D32; font-weight: 500;">Select Language / மொழியைத் தேர்ந்தெடுக்கவும்</div>
        <div style="text-align: center;">
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("English", key="btn_english"):
            st.session_state.current_language = "English"
            st.rerun()
    with col2:
        if st.button("Tamil", key="btn_tamil"):
            st.session_state.current_language = "Tamil"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Display Current Language
    st.markdown(f"""
    <div style="text-align: center; margin-top: 10px; padding: 10px; background: #E8F5E9; border-radius: 6px; color: #2E7D32; font-weight: 500;">
        Current Language: {st.session_state.current_language}
    </div>
    """, unsafe_allow_html=True)

    # Display chat history
    if not st.session_state.chat_history:
        if st.session_state.current_language == "English":
            st.markdown("""
            <div class="message bot-message">
                <div class="message-sender">AI Assistant</div>
                Welcome! I am your AI Agricultural Assistant. I can help you with government schemes, crop practices, weather information, market prices, pesticides, and crop diseases. Please select a language and ask your question.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="message bot-message">
                <div class="message-sender">AI உதவியாளர்</div>
                வணக்கம்! நான் உங்கள் AI விவசாய உதவியாளர். அரசு திட்டங்கள், பயிர் விவசாயம், வானிலை தகவல், சந்தை விலைகள், பூச்சிக்கொல்லிகள் மற்றும் பயிர் நோய்கள் குறித்து நான் உங்களுக்கு உதவ முடியும். தயவு செய்து ஒரு மொழியைத் தேர்ந்தெடுத்து உங்கள் கேள்வியைக் கேளுங்கள்.
            </div>
            """, unsafe_allow_html=True)

    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f"""
            <div class="message user-message">
                <div class="message-sender">You</div>
                {chat["message"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            sender = "AI Assistant" if st.session_state.current_language == "English" else "AI உதவியாளர்"
            st.markdown(f"""
            <div class="message bot-message">
                <div class="message-sender">{sender}</div>
                {chat["message"]}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Quick Questions Section
    st.markdown("""
    <div class="quick-questions-container">
        <div class="section-title">Frequently Asked Questions / அடிக்கடி கேட்கப்படும் கேள்விகள் (FAQ)</div>
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px;">
    """, unsafe_allow_html=True)

    current_lang = st.session_state.current_language
    questions = agriculture_kb[current_lang]["questions"][:15]  # Show first 15 questions

    for i, question in enumerate(questions):
        if st.button(question, key=f"quick_{i}"):
            # Add question to chat
            st.session_state.chat_history.append({
                "role": "user",
                "message": question
            })
            
            # Get answer
            answer = get_answer(question, current_lang)
            st.session_state.chat_history.append({
                "role": "assistant",
                "message": answer
            })
            
            # Update last question
            st.session_state.last_question = question
            st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)

    # User Input Section
    st.markdown("""
    <div class="input-container">
        <div class="section-title">Ask Your Question / உங்கள் கேள்வியைக் கேளுங்கள்</div>
    """, unsafe_allow_html=True)

    user_question = st.text_input(
        "",
        key="user_input",
        placeholder=f"Type your agricultural question in {st.session_state.current_language} here...",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Submit", key="submit_btn"):
            if user_question and user_question != st.session_state.last_question:
                # Add user question to chat
                st.session_state.chat_history.append({
                    "role": "user",
                    "message": user_question
                })
                
                # Get answer
                answer = get_answer(user_question, st.session_state.current_language)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "message": answer
                })
                
                # Update last question
                st.session_state.last_question = user_question
                st.rerun()

    with col2:
        if st.button("Clear Conversation", key="clear_btn"):
            st.session_state.chat_history = []
            st.session_state.last_question = ""
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Knowledge Areas Section
    st.markdown("""
    <div class="knowledge-container">
        <div class="section-title">Knowledge Areas Covered / அறிவு பகுதிகள்</div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="knowledge-column">
            <div class="knowledge-title">Government Schemes</div>
            <div class="knowledge-item">• PM-KISAN Scheme</div>
            <div class="knowledge-item">• Crop Insurance (PMFBY)</div>
            <div class="knowledge-item">• Kisan Credit Card</div>
            <div class="knowledge-item">• Soil Health Card</div>
            <div class="knowledge-item">• Farm Subsidies</div>
            <div class="knowledge-item">• Agricultural Loans</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="knowledge-column">
            <div class="knowledge-title">Crop Management</div>
            <div class="knowledge-item">• Sowing Times & Seasons</div>
            <div class="knowledge-item">• Fertilizer Recommendations</div>
            <div class="knowledge-item">• Irrigation Methods</div>
            <div class="knowledge-item">• Soil Fertility</div>
            <div class="knowledge-item">• Organic Farming</div>
            <div class="knowledge-item">• Crop Rotation</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="knowledge-column">
            <div class="knowledge-title">Problem Solving</div>
            <div class="knowledge-item">• Pest Control</div>
            <div class="knowledge-item">• Disease Management</div>
            <div class="knowledge-item">• Weather Advisory</div>
            <div class="knowledge-item">• Market Prices</div>
            <div class="knowledge-item">• Storage Solutions</div>
            <div class="knowledge-item">• Water Management</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>AI Agricultural Assistant | Support for Indian Farmers | Available 24/7</p>
        <p>Note: This assistant provides agricultural guidance. For critical decisions, consult local agriculture officers.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
