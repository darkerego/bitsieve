from datetime import datetime

import numpy as np
import paho.mqtt.client as mqtt
import random

from utils import mq_queue
#from ftxtool import mqtt_que
mqtt_que = mq_queue.MqttQue()
from utils import sql_lib
import pandas as pd


from trade_engine import kline_que
klines = {}

sql = sql_lib.SQLLiteConnection()

class MqSkel:
    def __init__(self, host='localhost', port=1883, debug=False):
        self.streamid = 'blackmirror_22'
        append = ''.join([x for x in random.sample('abcdefghijkl', 5)])
        self.streamid += append
        print(f'Streamid is {self.streamid}')

        self.host = host
        self.port = port
        self.debug = False
        self.CLIENTS = {}
        # SUBSCRIPTIONS = [("/incoming/" + v, 0)  for v in PAIRS]
        self.SUBSCRIPTIONS = [('/super_signals', 0)]
        self.mqStart()

    def mqConnect(self, client, userdata, flags, rc):
        """ MQTT Connect Event Listener
        :param client:      Client instance
        :param userdata:    Private userdata as set in Client() or userdata_set()
        :param flags:       Dict of broker reponse flags
        :param rc:          Int of connection state from 0-255:
                                0: Successful
                                1: Refused: Incorrect Protocol
                                2: Refused: Invalid Client ID
                                3: Refused: Server Unavailable
                                4: Refused: Incorrect User/Password
                                5: Refused: Not Authorised
        """
        if rc == 0:
            print("Connected Successfully")
        else:
            print("Refused %s" % rc)


    def mqDisconnect(self, client, userdata, rc):
        """ MQTT Connect Event Listener
        :param client:      Client instance
        :param userdata:    Private userdata as set in Client() or userdata_set()
        :param rc:          Int of disconnection state:
                                0: Expected Disconnect IE: We called .disconnect()
                                _: Unexpected Disconnect
        """
        if rc == 0:
            print("Disconnected")
        else:
            print("Error: Unexpected Disconnection")


    def mqParse(self, client, userdata, message):
        """
        1499040000000,      // Open time
            "0.01634790",       // Open
            "0.80000000",       // High
            "0.01575800",       // Low
            "0.01577100",       // Close
            "148976.11427815",  // Volume
            1499644799999,      // Close time
            "2434.19055334",    // Quote asset volume
            308,                // Number of trades
            "1756.87402397",    // Taker buy base asset volume
            "28.46694368",      // Taker buy quote asset volume
            "17928899.62484339" // Ignore.
          ]
        """

        if "/tickers/" in message.topic:

            candle = message
            kline_que_name = message.topic.split('/tickers/')[1]

            open_time = candle.get(kline_que_name).get('kline').get('open_time')
            open =  candle.get(kline_que_name).get('kline').get('open')
            low = candle.get(kline_que_name).get('kline').get('low')
            mid = candle.get(kline_que_name).get('kline').get('mid')
            high = candle.get(kline_que_name).get('kline').get('high')
            close = candle.get(kline_que_name).get('kline').get('close')
            base_volume = candle.get(kline_que_name).get('kline').get('base_volume')
            quote_volume = candle.get(kline_que_name).get('kline').get('quote_volume')
            klines.update(symbol_tf=kline_que_name, kline=message.payload)
            #sql.add_table(kline_que_name)
            #sql.append([[open_time, open, high, low, close, base_volume ]], table=kline_que_name)
            new_time = [datetime.fromtimestamp(time / 1000) for time in open_time]
           #  return open_time, low, mid, high, close, close_array, high_array, low_array, new_time, volume_array




            k_que = kline_que.KlineQue(message.topic)
            if self.debug:
                print('Command Data')
            # spreadParse(message.payload, message.topic)
        elif "/echo" in message.topic:
            if self.debug:
                print(message.payload)
        elif '/signals' in message.topic:
            mqtt_que.append(message.payload.decode())


    def mqPublish(self, payload, topic, qos=0, retain=False, id=None):
        """ MQTT Publish Message to a Topic
        :param id           String of the Client ID
        :param topic:       String of the message topic
        :param payload:     String of the message body
        :param qos:         Int of QoS state:
                                0: Sent once without confirmation
                                1: Sent at least once with confirmation required
                                2: Sent exactly once with 4-step handshake.
        :param retain:      Bool of Retain state
        :return             Tuple (result, mid)
                                result: MQTT_ERR_SUCCESS or MQTT_ERR_NO_CONN
                                mid:    Message ID for Publish Request
        """
        id = self.streamid
        global CLIENTS

        client = self.CLIENTS.get(id, False)
        if not client:
            raise ValueError("Could not find an MQTT Client matching %s" % id)
        client.publish(topic, payload=payload, qos=qos, retain=retain)


    def mqStart(self):
        """ Helper function to create a client, connect, and add to the Clients recordset
        :param streamID:    MQTT Client ID
        :returns mqtt client instance
        """
        global CLIENTS
        print(f'Connecting to {self.host}:{self.port}')
        client = mqtt.Client(self.streamid, clean_session=False)

        # client.username_pw_set(config.mq_user, config.mq_pass)
        # Event Handlers
        client.on_connect = self.mqConnect
        client.on_disconnect = self.mqDisconnect
        client.on_message = self.mqParse
        # Client.message_callback_add(sub, callback) TODO Do we want individual handlers?
        # Connect to Broker
        client.connect(self.host, port=self.port,
                       keepalive=60)
        # Subscribe to Topics
        client.subscribe(self.SUBSCRIPTIONS)  # TODO Discuss QoS States
        client.loop_start()
        self.CLIENTS[self.streamid] = client
        return client


