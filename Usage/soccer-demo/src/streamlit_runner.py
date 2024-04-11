import streamlit as st
import cv2
import socket
import struct
import pickle
import time
import select

st.set_page_config(layout="wide")

def attempt_connection(host, port, status_placeholder):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        client_socket.settimeout(5.0)  # Set a timeout for socket operations to detect disconnection
        status_placeholder.empty()  # Clear status message upon successful connection
        return client_socket
    except (ConnectionRefusedError, socket.timeout) as e:
        status_placeholder.text("Server is not available. Retrying...")
        time.sleep(2)
        return None

def clear_frames(placeholders):
    # Clear all frame placeholders to display empty frames
    for row in placeholders:
        for placeholder in row:
            placeholder.empty()

def check_socket_alive(sock):
    try:
        # Use select to check socket status with a non-blocking call
        ready = select.select([sock], [], [], 0.1)
        if ready[0]:  # Readable sockets indicate available data or disconnection
            # Peek to check if connection was closed without blocking
            if len(sock.recv(1, socket.MSG_PEEK)) == 0:
                return False
        return True
    except socket.error:
        return False

def receive_and_display_streams(host='127.0.0.1', port=8080):
    """
    Receives video frames from a server and displays them using Streamlit.

    Args:
        host (str, optional): The IP address or hostname of the server. Defaults to '127.0.0.1'.
        port (int, optional): The port number of the server. Defaults to 8080.
    """
    st.title("Soccer Analysis")

    status_placeholder = st.empty()  # Placeholder for showing connection status
    captions = ['Frame', 'Layout', 'Heatmap A', 'Heatmap B']

    # Initialize the grid for displaying video frames
    row1_cols = st.columns(2)
    row2_cols = st.columns(2)
    placeholders = [
        [row1_cols[0].empty(), row1_cols[1].empty()],
        [row2_cols[0].empty(), row2_cols[1].empty()]
    ]

    client_socket = None
    while True:
        if client_socket is None or not check_socket_alive(client_socket):
            if client_socket is not None:
                try:
                    client_socket.close()  # Attempt to close the previous connection if it exists
                except:
                    pass
            client_socket = attempt_connection(host, port, status_placeholder)
            if client_socket is None:
                continue

        data = b""
        payload_size = struct.calcsize("Q")

        try:
            while True:
                while len(data) < payload_size:
                    packet = client_socket.recv(4096)
                    if not packet:  # Connection closed by the server
                        raise Exception("Connection closed")
                    data += packet

                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q", packed_msg_size)[0]

                while len(data) < msg_size:
                    data += client_socket.recv(4096)

                frame_data = data[:msg_size]
                data = data[msg_size:]

                frames = pickle.loads(frame_data)
                for i, compressed_frame in enumerate(frames):
                    frame = cv2.imdecode(compressed_frame, cv2.IMREAD_COLOR)
                    if len(frame.shape) == 2:  # Convert grayscale images for display
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Determine the placeholder's row and column based on the frame index
                    row, col = divmod(i, 2)
                    # Update the corresponding placeholder
                    placeholders[row][col].image(frame, caption=captions[i], width=700)  # Adjust width as needed

        except Exception as e:
            client_socket.close()
            client_socket = None
            clear_frames(placeholders)  # Clear the frames when connection is lost or any error occurs

if __name__ == '__main__':
    receive_and_display_streams()
