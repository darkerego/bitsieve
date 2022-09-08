import hmac
import random
import string
from _hashlib import pbkdf2_hmac
from getpass import getpass
from os import urandom, mkdir
from typing import Tuple

from .aes_lib import HandleAes
from lib.clipw_conf import debug, config_dir, config_file


class Hash_pass(object):
    def __init__(self):
        pass


    def hash_new_password(self, password: str) -> Tuple[bytes, bytes]:
        """
        Hash the provided password with a randomly-generated salt and return the
        salt and hash to store in the database.
        """
        salt = urandom(16)
        pw_hash = pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        return salt, pw_hash

    def check_pw_padding(self, pw):
        """
        Check that password length is AES compliant, add padding if not
        :param pw: password
        :return: password with buffering
        """
        pw_len = len(pw)
        if pw_len <= 8:
            while pw_len % 8 != 0:
                pw += '0'
                pw_len = (len(pw))
            return pw
        if pw_len <= 16:
            while pw_len % 16 != 0:
                pw += '0'
                pw_len = (len(pw))
            return pw
        if pw_len <= 32:
            while pw_len % 32 != 0:
                pw += '0'
                pw_len = (len(pw))

    def is_correct_password(self, salt: bytes, pw_hash: bytes, password: str) -> bool:
        """
        Given a previously-stored salt and hash, and a password provided by a user
        trying to log in, check whether the password is correct.
        """

        return hmac.compare_digest(
            pw_hash,
            pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        )

    def generate_hash(self, pw):
        """
        Generate a Hash and Salt from a password
        :param self: password to hash
        :return: salt and hash
        """
        salt, pw_hash = self.hash_new_password()
        salt = salt.hex()
        pw_hash = pw_hash.hex()
        if self.is_correct_password(bytes.fromhex(salt), bytes.fromhex(pw_hash), pw):
            if debug:
                print('Test Succeeded!')
                print("Salt: %s" % salt)
                print("Hash: %s" % pw_hash)
            return salt, pw_hash
        else:
            return False

    def store_master_password(self):
        """
        Upon init, store users master key to disc
        :return: --
        """

        try:
            mkdir(config_dir)
        except FileExistsError as err:
            pass
        else:
            print('Created config directory %s' % config_dir)
        # hash a password
        while True:
            pw = getpass('Password: ')
            pw2 = getpass('Confirm: ')
            if pw == pw2:
                pw_len = len(pw)
                if pw_len > 32 or pw_len < 8:
                    print('Password must be at least 8 and no more than 32 characters.')
                    exit(1)
                else:
                    s, p = self.generate_hash(pw)
                    hash_str = str(s + ":" + p)
                    with open(config_file, 'w+') as f:
                        f.write(hash_str)
                    return True

            else:
                print('Passwords do not match. Try again')

    def get_master_password(self):
        """
        Function to create an AES friendly Master Password
        to encrypt all the passwords in the database...
        Because they key needs to be either 8 or 16 characters,
        we will add padding until it if an appropriate length.
        :return: byte encoded password string
        """
        with open(config_file, 'r') as ff:
            master_hash = ff.read()
            _salt = master_hash.split(':')[0]
            _hash = master_hash.split(':')[1]
            master_pw = getpass("Master Password: ")
            if self.is_correct_password(bytes.fromhex(_salt), bytes.fromhex(_hash), master_pw):
                print('Success!')
                master_pw = self.check_pw_padding(master_pw)
                return master_pw.encode()
            else:
                print('Incorrect Password')
                exit(1)

    def store_password(self):
        """
        Function to encrypt a password before inserting into database
        :param master_key:
        :return:
        """

        while True:
            pw = getpass('Password: ')
            pw2 = getpass('Confirm: ')
            if pw == pw2:

                return pw

    def random_password(self, n: int = 12) -> object:
        """
        Generate a random password of length n
        :param n: length of password to generate
        :return: password string
        """
        random_source = string.ascii_letters + string.digits + string.punctuation
        password = random.choice(string.ascii_lowercase)
        password += random.choice(string.ascii_uppercase)
        password += random.choice(string.digits)
        password += random.choice(string.punctuation)

        for i in range(n):
            password += random.choice(random_source)

        password_list = list(password)
        random.SystemRandom().shuffle(password_list)
        password = ''.join(password_list)
        return password
