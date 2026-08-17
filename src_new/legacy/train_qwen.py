from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def test_qwen_brain():
    print("=== 正在將 Qwen-3B 載入 RTX 4090 顯示卡中 (請稍候) ===")
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    
    # 載入模型與分詞器，device_map="auto" 會自動把模型放到 GPU 上
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype="auto", 
        device_map="auto"
    )
    print("✅ 載入完成！開始進行空間邏輯測試...\n")

    # 這是我們模擬 YOLO 抓到的座標，設計一個情境給大腦
    prompt = """
    當前桌面的物件座標如下 (X軸越小越左邊，Y軸越小越靠近第一排)：
    編號 0: 紅色積木, 座標 (150, 200)
    編號 1: 藍色積木, 座標 (350, 200)
    編號 2: 綠色積木, 座標 (150, 400)
    
    使用者的指令是：「幫我拿第一排右邊的那個積木」
    請分析 Y 軸判斷排數，X 軸判斷左右。目標是哪一個編號？請只回答數字。
    """
    
    # 依照 Qwen 官方的對話格式封裝指令
    messages = [
        {"role": "system", "content": "你是一個精準的機器人空間邏輯大腦，只能輸出數字。"},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    print(f"[傳送給 Qwen 的提示詞]:\n{prompt}")
    print("-" * 40)
    print("思考中...")
    
    # 執行推論 (這就是榨乾 4090 算力的一刻)
    generated_ids = model.generate(**model_inputs, max_new_tokens=20)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"\n🧠 [Qwen 的回答]: {response}")
    print("\n(正確答案應該是 1，因為編號 0 和 1 的 Y 軸都是 200 屬於第一排，而編號 1 的 X 軸 350 比較靠右)")

if __name__ == "__main__":
    test_qwen_brain()