import paho.mqtt.client as mqtt

streamid = 'bot_test'
CLIENTS = {}
SUBSCRIPTIONS = [('/incoming', 0)]
tickers = {}
from utils import m_que
from utils.colorprint import NewColorPrint

mq_q = m_que
cp = NewColorPrint()
DEBUG = False

class Mqtt:

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
        if "/incoming/" in message.topic:
            command = message.topic.replace("/command/", "")
            if DEBUG: print('Command Data')
            #spreadParse(message.payload, message.topic)
        elif "/echo" in message.topic:
            if DEBUG: print(message.payload)
            """try:
                TickerParse(message)
            except Exception as err:
                if debug:
                    print('Mq Parse Error:' + str(err))"""


    def mqPublish(self, client_id, payload, topic, qos=0, retain=False):
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
        id = client_id
        global CLIENTS
        client = CLIENTS.get(id, False)
        if not client:
            raise ValueError("Could not find an MQTT Client matching %s" % id)
        client.publish(topic, payload=payload, qos=qos, retain=retain)


    def mqStart(self, streamId):
        """ Helper function to create a client, connect, and add to the Clients recordset
        :param streamID:    MQTT Client ID
        :returns mqtt client instance
        """
        global CLIENTS
        client = mqtt.Client(streamId, clean_session=False)
        # client.username_pw_set(config.mq_user, config.mq_pass)
        # Event Handlers
        client.on_connect = self.mqConnect
        client.on_disconnect = self.mqDisconnect
        client.on_message = self.mqParse
        # Client.message_callback_add(sub, callback) TODO Do we want individual handlers?
        # Connect to Broker
        client.connect('ws.blackmirror.app', port=1883,
                       keepalive=60)
        # Subscribe to Topics
        client.subscribe(SUBSCRIPTIONS)  # TODO Discuss QoS States
        client.loop_start()
        CLIENTS[streamId] = client
        return client

    client = mqStart('blackmirror')