# type: ignore
import socket


class Socket:
    """
    Handles the barest level of UDP communication with a server in a slightly
    simpler way (for our purposes) than the default socket library.
    """

    def __init__(self, host, port, bufsize=8192):
        """
        host: hostname of the server we want to connect to
        port: port of the server we want to connect to
        """

        self.address = (host, port)
        self.bufsize = bufsize

        # the socket communication with the server takes place on (ipv4, udp)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, msg, append_null_terminator=True):
        """
        Sends a message to the server.  Appends a null terminator by default.
        """

        # append a null terminator if requested
        if append_null_terminator:
            msg = msg + "\0"

        self.sock.sendto(str.encode(msg), self.address)

    def recv(self, conform_address=True):
        """
        Receives data from the given socket.  Returns the data as a string.
        If conform_address is True, the address the server sent its response
        from replaces the address and port set at object creation.
        """

        data, address = self.sock.recvfrom(self.bufsize)

        if conform_address:
            self.address = address

        return data
