import hashlib
from unidecode import unidecode


def create_author_id(fullname, pub_id):
    return "00|author______::" + hashlib.md5(pub_id.encode("utf-8") + fullname.encode("utf-8")).hexdigest()


def lnfi(author):
    try:
        return (unidecode(author['surname'] + author['name'][0])).lower().strip()
    except:
        return ""


def is_author_wellformed(author):
    return author['name'] != "" and author['surname'] != ""


def replace_char(s, pos, c):
    s = list(s)
    s[pos] = c
    return "".join(s)
