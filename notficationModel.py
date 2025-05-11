from notifypy import Notify

def PushNotification(noti: str) -> None:

    notification = Notify()
    notification.application_name = " "
    notification.title = "Hey!"
    notification.message = noti
    notification.icon = ""
    notification.send()


PushNotification("Are you still there?")
