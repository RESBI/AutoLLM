#
#    AutoLLM.py
#    Main goal: 
#     LLM server (Ollama, Deepseek, Siliconflow, OpenAI, etc.)
#      ||   /\
#      ||   ||
#      ||   ||   *KEEP SLEFCHECKING
#      ||   ||
#      \/   ||
#     AutoLLM.py
#         /\
#         ||
#         ||  Interrupt selfchecking, and send the result to the user.
#         ||  Sometimes AutoLLM will raises a topic automatically. 
#         ||
#         \/
#        User


from openai import OpenAI
import json
import time
import shutil
import os


input_dir = "./inputs"
processed_dir = "./processed"
output_dir = "./outputs"
memory_dir = "./memory"

recv_msg_pattern = "现在是<<<time>>>, <<<user_name>>> 给你发来一封信，说：\"<<<user_message>>>\"，请你在文末严格以格式：<letter>{\"name\":\"<<<user_name>>>\", \"title\":\"[这里写上回复的标题]\", \"content\":\"[这里是你的回复内容]\", \"time\":\"[这里写上现在的时间]\"}</letter>, 并在中括号中按照提示填写相应的内容。"
mimic_prompt = "以上是你之前的想法。你是一个有着高中青春心态的安静的少女, 你有着多愁善感的情感, 也总是希望着事情变好。现在是<<<time>>>, 请详细总结已经收到的消息, 并判断如果作为一个人类, 你现在应该是什么情绪? 如果如果推断结论是你需要对收到过来信的某个人发消息, 请在文末以格式<letter>{\"name\":\"[名字]\", \"title\":\"[这里写上回复的标题]\", \"content\":\"[这里是你的消息内容]\", \"time\":\"[这里写上现在的时间]\"}</letter>, 并在中括号中按照提示填写相应的内容。如果推断结论是不需要给谁发消息, 则只需要总结, 并分析情绪即可"

api_key_file = "api_key.txt"
with open(api_key_file, "r") as api_key_reading: 
    api_key = api_key_reading.read()


def make_timestamp():
    return time.strftime("%Y/%m/%d %H:%M:%S UTC%z %a", time.gmtime())


def make_message(timestamp, user_name, user_message, recv_msg_pattern=recv_msg_pattern):
    return recv_msg_pattern.replace("<<<time>>>", timestamp).replace("<<<user_name>>>", user_name).replace("<<<user_message>>>", user_message)


def make_mimicing_message(timestamp, mimic_prompt=mimic_prompt):
    return mimic_prompt.replace("<<<time>>>", timestamp)


def get_llm_response(content, api_key=api_key):
    client = OpenAI(
        api_key=api_key, 
        base_url="https://api.siliconflow.cn/v1",
        timeout=1200, # in seconds
    )

    response = client.chat.completions.create(
        model='deepseek-ai/DeepSeek-R1',
        messages=[
            {'role': 'user', 
            'content': content}
        ],

        stream=True
    )

    result = ""

    for chunk in response:
        result += str(chunk.choices[0].delta.content)

    return result.strip("None")


def pick_message(input_dir):
    # pick one, the first one. 
    for file in os.listdir(input_dir):
        with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
            content = f.read()
            f.close()
            # move it to processed
            shutil.move(os.path.join(input_dir, file), os.path.join(processed_dir, file))
            return content
    return ""


def parse_letter(llm_response):
    # find the first <letter> and the last </letter>
    start = llm_response.find("<letter>")
    end = llm_response.find("</letter>")
    return llm_response[start + 8 : end]


def AutoLLM(): 
    print("AutoLLM starts.")
    print("[AutoLLM] Checking directories...")
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    if not os.path.exists(memory_dir):
        os.makedirs(memory_dir)

    print("[AutoLLM] Loading memory...")
    # Get memory 
    try:
        memory_file = open(os.path.join(memory_dir, "memory.txt"), "r", encoding="utf-8")
        memory = memory_file.read()
        memory_file.close()
    except:
        memory = ""

    print("[AutoLLM] Start!")
    while True:
        # pick one message
        user_message = pick_message(input_dir)
        llm_success = False

        while not llm_success:
            current_timestamp = make_timestamp()
            print("[AutoLLM] Try to contact LLM...{}".format(current_timestamp))

            user_message_content = ""
            if user_message != "":
                # make a message
                user_message_content = make_message(current_timestamp, "Resbi", user_message)
                # focus on replying user's message
                to_llm_message = memory + "\n\n" + user_message_content
            else: 
                to_llm_message = memory + "\n\n" + make_mimicing_message(current_timestamp)

            try:
                llm_response = get_llm_response(to_llm_message)
            except:
                llm_response = ""

            if llm_response != "":
                llm_success = True
                print("[AutoLLM] Got response!")
                print(llm_response)
                # save the history
                print("[AutoLLM] Saving history...")
                with open(os.path.join(memory_dir, "history.txt"), "a", encoding="utf-8") as f:
                    f.write(to_llm_message + "\n\n" + llm_response)

                # update AutoLLM's memory
                print("[AutoLLM] Saving memory...")
                with open(os.path.join(memory_dir, "memory.txt"), "a", encoding="utf-8") as f:
                    f.write(llm_response)
                
                # if it sends a letter...
                letter = parse_letter(llm_response)
                if letter != "":
                    # send a letter
                    print("[AutoLLM] sends a letter:")
                    print(letter)
                    try:
                        letter_json = json.loads(letter)
                        with open(os.path.join(output_dir, letter_json["name"] + str(time.time()) + ".txt"), "w", encoding="utf-8") as f:
                            f.write(letter_json["content"])
                    except:
                        with open(os.path.join(output_dir, str(time.time()) + ".txt"), "w", encoding="utf-8") as f:
                            f.write(letter)



if __name__ == "__main__":
    #print(get_llm_response(make_message(make_timestamp(), "Resbi", "你好~")))
    print("[AutoLLM] Using api_key: {}".format(api_key))
    AutoLLM()


