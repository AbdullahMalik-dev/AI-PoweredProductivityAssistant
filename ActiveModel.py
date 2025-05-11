from pynput import mouse, keyboard
import time
key_count = 0
mouse_click_count = 0
def Picks() -> None:
    def on_press(key):
        global key_count
        key_count += 1

    # Mouse listener callback for clicks
    def on_click(x, y, button, pressed):
        global mouse_click_count
        if pressed:
            mouse_click_count += 1

    keyboard_listener = keyboard.Listener(on_press=on_press)
    mouse_listener = mouse.Listener(on_click=on_click)

    keyboard_listener.start()
    mouse_listener.start()

    start_time = time.time()
    tracking_duration = 5
    while time.time() - start_time < tracking_duration:
        time.sleep(1)
    
    return mouse_click_count, key_count

def Listening():

    def any_key_pressed():
        pressed = False

        def on_press(_):

            nonlocal pressed
            pressed = True
            return False
        
        with keyboard.Listener(on_press=on_press) as listener:
            time.sleep(2.5)
        return pressed

    
    return any_key_pressed()

def attentionTypingRate(inTime):
    rate = [2]
    timer = inTime
    start_time = time.time()
    end_time = 0
    for i in range(timer):
        if Listening() == True:
            end_time = time.time()
        else:
            end_time = start_time
        time.sleep(0.5)
        finalTime = end_time - start_time
        rateCalc = min(100,(round(finalTime) / 100))
        rate.append(rateCalc)
        if len(rate) == 3:
            rate.pop(0)
    return rate

def isActive():
    newRate = 0
    M,K = 0,0
    focusRate = 0
    prev_rate = attentionTypingRate(4)
    M, K = Picks()
    newRate = prev_rate[-1] + prev_rate[-2]
    focusRate = (100 - (M + K + newRate))
    if (M+K) == 0 or focusRate <= 0:
       focusRate = int(newRate * 100)
    if focusRate != 0:
        focusRate = 100 - focusRate
        if focusRate > 90:
            focusRate = 0
    return focusRate
