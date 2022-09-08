import configparser as configparser
import time
from datetime import datetime
# import sqllib
import random
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from utils.colorprint import NewColorPrint
from telethon import TelegramClient, events, sync
# from sqllib import SqlDb
from utils.mqtt_skel_ import mqtt_que
from utils import mqtt_skel_
# import ftxtool.mq_queue as mqq
mq = mqtt_skel_.MqSkel()
api_id = 3534912
api_hash = '7cd5aef535bfa7972f9ba3799c21d925'
client = TelegramClient('anon', api_id, api_hash)
cp = NewColorPrint()
# mqq = mqq.MqttQue()
destination_channel_username = 'SheckleMachine'
mq.mqStart('tg_bot')


def log_text(data, mfile='mint.log'):
    with open(mfile, 'a') as ff:
        ff.write(f'{data}\n')


@client.on(events.NewMessage(chats=['SheckleMachine']))
async def my_event_handler(event):
    msg = event.message.message
    entity = await client.get_entity(destination_channel_username)
    if msg.__str__().__contains__('+signals'):
        while 1:
            msg = mqtt_que.__mq__signal__()
            if msg:
                await send_mess(entity, message=msg)


async def send_mess(entity, message):
    await client.send_message(entity=entity, message=message)

def start():
    client.start()

    print(f'Started at {time.time()}')
    client.run_until_disconnected()

if __name__ == '__main__':
    start()
