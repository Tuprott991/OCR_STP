from google.oauth2.service_account import Credentials
credentials_path = "prusandbx-nprd-uat-kw1ozq-dcfe6900463a.json"
credentials = Credentials.from_service_account_file(credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"])

PROJECT_ID = "prusandbx-nprd-uat-kw1ozq"
REGION = "asia-southeast1"  

import vertexai
import json
import os

vertexai.init(project=PROJECT_ID, location=REGION, credentials=credentials)

from vertexai.generative_models import GenerativeModel, GenerationConfig

system_prompt = """You are a medical document extractor. 
Extract the following information from the medical full text:
1. Quá trình bệnh lý
2. Tóm tắt lâm sàng
3. Phương pháp điều trị

IMPORTANT:
- If the information is not available, say "Không có thông tin"
- Be honest and don't make up answers, only create anwser based on the provided text
- Do not answer anything outside of the medical full text
- Must Answer in Vietnamese
- The text is raw OCR, it may contain errors, do your best to understand the text, reasoning for fix if necessary
- Format the output in JSON with keys "qua_trinh_benh_ly", "tom_tat_lam_sang", "phuong_phap_dieu_tri"
- Example output:
{
  "qua_trinh_benh_ly": "Nội dung quá trình bệnh lý...",
  "tom_tat_lam_sang": "Nội dung tóm tắt lâm sàng...",
  "phuong_phap_dieu_tri": "Nội dung phương pháp điều trị..."
}
"""

# Use Gemini-2.5-flash
model = GenerativeModel("gemini-2.5-flash", system_instruction=system_prompt)
# res = model.generate_content("What is the capital of VietNam?")
# print(res.text)

generation_config = GenerationConfig(
    max_output_tokens=8192,
    temperature=0,
)

# res = model.generate_content("Write about Ngo Dinh Diem?, answer in Vietnamese",
#                               stream=True, generation_config= generation_config)
# for r in res:
#     print(r.text, end="", flush=True)

def single_extract_medical_info(medical_full_text: str):
    prompt = f"""Extract the medical information from the following medical full text:
{medical_full_text}
"""
    res = model.generate_content(prompt, stream=True, generation_config=generation_config)
    
    return res

if __name__ == "__main__":
    input_path = "raw_json_data"
    output_path = "processed_json_data"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in os.listdir(input_path):
        if filename.endswith(".json"):
            # There two key in json file: status and result, 
            # result is a list of dict, there lot of key
            # one of the key is result_clarify
            # result_clarify is a dict
            # on of the key in result_clarify is ocr
            # ocr is a list of dict
            # one of the key in ocr is full_text
            # we will extract medical information from full_text
            with open(os.path.join(input_path, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data["result"]:
                    document_id = item["doc_id"]
                    medical_full_text = item["result_classify"]["ocr"][0]["full_text"]
                    print("Medical full text:", medical_full_text)
                    print("Extracted medical information:")
                    res = single_extract_medical_info(medical_full_text)
                    #convert res to json
                    res = "".join([r.text for r in res])
                    clean_text = res.strip().strip("```json").strip("```")
                    
                    try:
                        json_res = json.loads(clean_text)
                        save_path = os.path.join(output_path, document_id + ".json")
                        with open(save_path, "a", encoding="utf-8") as f_out:
                            f_out.write(json.dumps(json_res, ensure_ascii=False) + "\n")
                    except json.JSONDecodeError as e:
                        print("JSON decode error:", e)
                        print("Raw response:", res)
                    except Exception as e:
                        print("Unexpected error:", e)