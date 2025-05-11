import psutil
import time
import numpy as np
import logging
import yake
from stable_baselines3 import PPO
from transformers import BertTokenizer, BertModel, pipeline
import torch
import warnings
import gymnasium as gym
from gymnasium import spaces
from notificationModel import PushNotification
from ActiveModel import isActive
import multiprocessing
from inputimeout import inputimeout

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#? # Class for generating study guide summaries Guided by OpenAI's ChatGPT
class StudyGuideAI:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased").to(device)
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
        self.kw_extractor = yake.KeywordExtractor()

#* Function to extract key concepts (keywords) from the input text.
    def extract_key_concepts(self, text):
        keywords = self.kw_extractor.extract_keywords(text)
        key_concepts = [kw[0] for kw in keywords[:5]]
        return key_concepts
    
#* Function to generate a study guide by summarizing the text and extracting key concepts.
    def generate_study_guide(self, text):
        summary = self.summarizer(text, max_length=min(len(text.split()), 150), min_length=20, do_sample=False)
        key_concepts = self.extract_key_concepts(text)
        return {"summary": summary[0]['summary_text'], "key_concepts": key_concepts}


#? Creating the Environment Guided by OpenAI's ChatGPT
class StudyEnv(gym.Env):
    def __init__(self):
        super(StudyEnv, self).__init__()
        self.state = 100
        self.done = False

        self.observation_space = spaces.Box(low=np.array([0, -100]), high=np.array([100, 100]), shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def step(self, action):
        prev_focus = self.state
        self.prev_focus = prev_focus
        self.focus_drop = 5

        if action == 0:
            self.state = max(0, self.state - np.random.randint(5, 15))
        elif action == 1:
            self.state = min(100, self.state + np.random.randint(10, 30))
        elif action == 2:
            self.state = min(100, self.state + np.random.randint(5, 20))

        if self.prev_focus >= 80 and action == 0:
            reward = 1  
        elif self.prev_focus < 70 and action == 1:
                reward = 1
        elif 70 <= self.prev_focus < 80 and action == 1:
            reward = 1
        elif 70 <= self.prev_focus < 80 and action == 2:
                reward = -1
        elif -1 <= self.focus_drop > 15 and action != 1:
            reward = -1
        elif self.focus_drop == 0 and action == 1:
            reward = 1
        else:
                reward = -0.5

        self.done = self.state == 100
        obs = np.array([self.state], dtype=np.float32)

        return obs, reward, self.done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 100
        self.prev_focus = 100
        self.focus_drop = 5
        self.done = False
        return np.array([self.state], dtype=np.float32), {}

    def render(self):
        print(f"Focus level: {self.state}")
        print(f"Previous Focus Level: {self.prev_focus}")

    def monitor_screen_usage(self):
        app_usage = {}
        start_time = time.time()
        while time.time() - start_time < 30:
            for process in psutil.process_iter(attrs=['pid', 'name']):
                name = process.info['name']
                app_usage[name] = app_usage.get(name, 0) + 1
            time.sleep(1)
        return sorted(app_usage.items(), key=lambda x: x[1], reverse=True)[:5]
    

#?   End of StudyEnv Class Function



##?     Start of the Code executer
if __name__ == '__main__':

    #? Just some random Lists
    processes = []  # * Using it for the process that are being used currently

    env = StudyEnv()
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=5000)

    focus_history = [1,1,1,1,1]
    max_history_length = 5  

    while True:
        print("-----------------------")

        #! Featues to add:
        #TODO Add Daily Reflection Prompt End-of-day AI-generated summary with prompts like: “What helped you focus today?”, “What will you try tomorrow?”
        #* Done making it loop, and give right output
        #* Done Make Different action outputs
        #* Done Text Summarizer
        #* Done AFK Detection
        #* Done Notifier

        if len(focus_history) > max_history_length:
            focus_history.pop(0)
        if len(focus_history) >= max_history_length:
             focus_drop = focus_history[-2] - focus_history[-1]
        else:
            focus_drop = 0

        env.state = round(isActive()) #? Detects if user is active on the computer through reading keystrokes and mouse button clicks (its a custom function)
        env.focus_drop = focus_drop
        focus_history.append(env.state)
        obs = np.array([env.state, focus_drop], dtype=np.float32)

        def text(): #?  The Text Summarizer
            try: 
                text = inputimeout(prompt='Text to analyze for key concepts: ', timeout=5)
                if "shutquit" != text:
                    study_ai = StudyGuideAI()
                    guide = study_ai.generate_study_guide(text)

                    print("\nStudy Guide Summary:\n", guide["summary"])
                    print("\nKey Concepts in this Text:")
                    for concept in guide["key_concepts"]:
                        print("-", concept)
                else:
                    pass
            except:
                pass
        text()
        
        def action(): #? Predict the best action using the trained model 
            action, _ = model.predict(obs, deterministic=True)
            return action
        
        if len(focus_history) >= max_history_length:
            # p1 = multiprocessing.Process(target=action) #! Currently in the Process of trying to get multiprocessing to work

            action = action()
            screen_usage = env.monitor_screen_usage() #* checks the Processes running
            obs, reward, done, _, _ = env.step(action)
            actions_map = {0: "Keep Studying", 1: "Take a Break", 2: "Adjust Study Strategy"}
            decision = actions_map[int(action)]
            print("AI Decision:", decision)
            screen_usage.pop(0)
            processes.append(screen_usage)

            if done:
                obs, _ = env.reset()

        try:
            if processes[-1] != processes[-2]:
                print("\nTop Apps Used:", screen_usage)
        except: 
            pass
        ##? Notification
        if obs[0] < 10:
            PushNotification("Are you still there?")
        time.sleep(5)
