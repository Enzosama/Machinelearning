import ipaddress as ipaddress
from _typeshed import Unused

from psycopg2._psycopg import QuotedString, connection, cursor

def register_ipaddress(conn_or_curs: connection | cursor | None = None) -> None: ...
def cast_interface(s: str, cur: Unused = None) -> ipaddress.IPv4Interface | ipaddress.IPv6Interface | None: ...
def cast_network(s: str, cur: Unused = None) -> ipaddress.IPv4Network | ipaddress.IPv6Network | None: ...
def adapt_ipaddress(obj: object) -> QuotedString: ...
