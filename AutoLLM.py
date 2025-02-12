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
from win11toast import toast


input_dir = "./inputs"
processed_dir = "./processed"
output_dir = "./outputs"
memory_dir = "./memory"

recv_msg_pattern = "现在是<<<time>>>, <<<user_name>>> 给你发来一封信, 说：\"<<<user_message>>>\", 请你在文末严格以格式：<letter>{\"name\":\"<<<user_name>>>\", \"title\":\"[这里写上回复的标题]\", \"content\":\"[这里是你的回复内容]\", \"time\":\"[这里写上现在的时间]\"}</letter>, 并在中括号中按照提示填写相应的内容。除了回复的消息两侧, 请不要在其他的地方出现<letter>和</letter>。"
mimic_prompt = "以上是你之前的想法和内心经历。你是一个有着青春心态的安静的少女, 你多愁善感, 内心活动比较丰富, 表达总是富有想象但直白准确的, 也总是希望着事情能变得更好。请以这样的视角, 尽可能详细地记录上文中发生过的事要, 并以人类的视角分析情感, 不要使用太fancy的辞藻。现在是<<<time>>>, 请详细总结上文, 并判断如果作为一个人类, 你现在有什么内心活动？内心应该是什么情绪? 有什么要紧的事件? 有没有解决？还要做什么？你只需要对上文中出现过的对你发过消息的人回复消息。如果推断结论为你需要对你收到过来信的某个人发消息, 或者还未回复过某条消息, 请在文末以格式<letter>{\"name\":\"[名字]\", \"title\":\"[这里写上回复的标题]\", \"content\":\"[这里是你的消息内容]\", \"time\":\"[这里写上现在的时间]\"}</letter>, 并在中括号中按照提示填写相应的内容。除了回复的消息两侧, 请不要在其他的地方出现<letter>和</letter>。。如果推断结论是不需要给谁发消息, 或者已经回复过了, 则不需要在文末加上额外的东西, 只需要总结前文, 以短文的形式呈现内心活动, 并分析情绪即可。"

api_key_file = "api_key.txt"
with open(api_key_file, "r") as api_key_reading: 
    api_key = api_key_reading.read()


def make_timestamp():
    return time.strftime("%Y/%m/%d %H:%M:%S UTC%z %a", time.localtime())


def fill_message(timestamp, user_name, user_message, recv_msg_pattern=recv_msg_pattern):
    return recv_msg_pattern.replace("<<<time>>>", timestamp).replace("<<<user_name>>>", user_name).replace("<<<user_message>>>", user_message)


def fill_mimicing_message(timestamp, mimic_prompt=mimic_prompt):
    return mimic_prompt.replace("<<<time>>>", timestamp)



def get_llm_response(content, api_key=api_key):
    client = OpenAI(
        api_key=api_key, 
        base_url="https://api.siliconflow.cn/v1",
        timeout=1200, # in seconds
    )

    response = client.chat.completions.create(
        model='deepseek-ai/DeepSeek-R1',
        #model='Qwen/QVQ-72B-Preview',
        #model='deepseek-ai/DeepSeek-R1-Distill-Llama-70B', 
        #model='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', 
        messages=[
            {'role': 'user', 
            'content': content}
        ],

        stream=True
    )

    result = ""
    token_num = 0

    for chunk in response:
        token_num += 1
        result += str(chunk.choices[0].delta.content)

    return [token_num, result.strip("None")]


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


def update_memory(
        memory, 
        user_message, 
        user_message_content, 
        llm_response, 
        memory_dir
    ):
    # Save the memory.
    with open(os.path.join(memory_dir, "memory.txt"), "a", encoding="utf-8") as f:
        if user_message != "":
            memory += user_message_content + llm_response + "\n\n"
            f.write(user_message_content + llm_response + "\n\n")
        else: 
            memory += llm_response + "\n\n"
            f.write(llm_response  + "\n\n")
    return memory


def AutoLLM(MEM_LENGTH=4096, SLEEP_DURATION=12*60): 
    print("[AutoLLM] Start.")
    print("[AutoLLM] Only use the last {} characters of memory.".format(MEM_LENGTH))
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
            print("[AutoLLM] Trying to contact LLM...{}".format(current_timestamp))

            user_message_content = ""
            if user_message != "":
                # make a message
                user_message_content = fill_message(current_timestamp, "Resbi", user_message)
                # reply user's message and mimic
                to_llm_message = memory[-MEM_LENGTH:] + "\n\n" + user_message_content + "\n\n" + fill_mimicing_message(current_timestamp)
            else: 
                # just mimic
                to_llm_message = memory[-MEM_LENGTH:] + "\n\n" + fill_mimicing_message(current_timestamp)

            try:
                token_num, llm_response = get_llm_response(to_llm_message)
            except:
                llm_response = ""

            if llm_response != "":
                llm_success = True
                print("[AutoLLM] Got response! {} tokens...".format(token_num))
                print(llm_response)
                # save the history
                print("[AutoLLM] Saving history...")
                with open(os.path.join(memory_dir, "history.txt"), "a", encoding="utf-8") as f:
                    f.write(to_llm_message + "\n\n" + llm_response + "\n\n")

                # update AutoLLM's memory
                print("[AutoLLM] Saving memory...")
                memory = update_memory(memory, user_message, user_message_content, llm_response, memory_dir)
                print("[AutoLLM] Memory length: {} characters".format(len(memory)))


                # if it sends a letter...
                if "<letter>" in llm_response:
                    letter = parse_letter(llm_response)
                    if letter != "":
                        # send a letter
                        print("[AutoLLM] Got a letter! ")
                        print(letter)
                        try:
                            letter_json = json.loads(letter)
                            with open(os.path.join(output_dir, letter_json["name"] + str(time.time()) + ".txt"), "w", encoding="utf-8") as f:
                                f.write(letter_json["content"])
                            # create a notification
                            toast(
                                letter_json["title"],
                                letter_json["content"]
                            )
                        except:
                            with open(os.path.join(output_dir, str(time.time()) + ".txt"), "w", encoding="utf-8") as f:
                                f.write(letter)

        for sleeping_time in range(SLEEP_DURATION):
            if len(os.listdir(input_dir)) > 0:
                print("[AutoLLM] Found new messages! Respond it after 10 seconds...")
                time.sleep(10)
                break
            print("[AutoLLM] Sleeping... {} / {} seconds".format(sleeping_time, SLEEP_DURATION), end="\r")
            time.sleep(1)


if __name__ == "__main__":
    #print(get_llm_response(make_message(make_timestamp(), "Resbi", "你好~")))
    print("[AutoLLM] Using api_key: {}".format(api_key))
    AutoLLM()
