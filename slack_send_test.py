from knockknock import slack_sender


@slack_sender(webhook_url="webhook", channel="Vivek Anand", user_mentions=["U03DY5GB464"])
def main():
    even_arr = []
    for i in range(10000):
        if i%2==0:
            even_arr.append(i)

if __name__=='__main__':
    main()