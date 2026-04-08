import win32com.client.gencache
import time
import os
import glob
import socket


def log_msg(message: str):
    """
    Log a message with a timestamp.

    :param message: String, the message to be logged.
    :return: None
    """
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{current_time}] {message}")


class BmgCom:
    """
    A class to handle communication with the BMG SPECTROstar Nano plate reader using ActiveX.

    This class provides methods to control the plate reader, such as opening connections, running protocols,
    setting temperature, and inserting/ejecting plates.

    :param control_name: Optional name of the device to connect to. If provided, an attempt to open
                         a connection is made during initialization.
    """

    def __init__(self, control_name: str = None):
        """
        Initialize the BmgCom class and create the ActiveX COM object.

        :param control_name: Optional string specifying the control name for the reader. If provided,
                             an automatic connection is attempted.
        :raises: Exception if the ActiveX COM object instantiation or connection fails.
        """
        try:
            # Initialize ActiveX COM object
            self.com = win32com.client.gencache.EnsureDispatch("BMG_ActiveX.BMGRemoteControl")
            log_msg("COM object created successfully.")

        except Exception as e:
            log_msg(f"Instantiation failed: {e}")
            raise

        if control_name:
            self.open(control_name)

    def open(self, control_name: str):
        """
        Open a connection to the BMG reader.

        :param control_name: String specifying the name of the reader to connect to.
        :raises: Exception if the connection fails or returns an error status.
        """
        try:
            result_status = self.com.OpenConnectionV(control_name)
            if result_status:
                raise Exception(f"OpenConnection failed: {result_status}")
            log_msg(f"Connected to {control_name} successfully.")
        except Exception as e:
            log_msg(f"Failed to open connection: {e}")
            raise

    def version(self):
        """
        Retrieve the software version of the BMG ActiveX Interface.

        :return: String representing the software version.
        :raises: Exception if the version retrieval fails.
        """
        try:
            version = self.com.GetVersion()
            log_msg(f"Software version: {version}")
            return version
        except Exception as e:
            log_msg(f"Failed to get version: {e}")
            raise

    def status(self):
        """
        Get the current status of the plate reader e.g., 'Ready', 'Busy'.

        :return: String representing the current status of the reader.
        :raises: Exception if the status retrieval fails.
        """
        try:
            status = self.com.GetInfoV("Status")
            return status.strip() if isinstance(status, str) else 'unknown'
        except Exception as e:
            log_msg(f"Failed to get status: {e}")
            raise

    def temp1(self):
        """
        Get the current temperature of the incubator at the bottom heating plate.

        :return: String representing the current temperature of the bottom heating plate.
        :raises: Exception if the temp1 retrieval fails.
        """
        try:
            temp1 = self.com.GetInfoV("Temp1")
            return temp1.strip() if isinstance(temp1, str) else 'unknown'
        except Exception as e:
            log_msg(f"Failed to get Temp1: {e}")
            raise

    def temp2(self):
        """
        Get the current temperature of the incubator at the top heating plate.

        :return: String representing the current temperature of the top heating plate.
        :raises: Exception if the temp2 retrieval fails.
        """
        try:
            temp2 = self.com.GetInfoV("Temp2")
            return temp2.strip() if isinstance(temp2, str) else 'unknown'
        except Exception as e:
            log_msg(f"Failed to get Temp2: {e}")
            raise

    def plate_in(self):
        """
        Insert the plate holder into the reader.

        :return: None
        :raises: Exception if the plate insertion command fails.
        """
        try:
            self.exec(['PlateIn'])
            log_msg("Plate inserted into the reader.")
        except Exception as e:
            log_msg(f"Failed to insert plate: {e}")
            raise

    def plate_out(self):
        """
        Eject the plate holder from the reader.

        :return: None
        :raises: Exception if the plate ejection command fails.
        """
        try:
            self.exec(['PlateOut'])
            log_msg("Plate ejected from the reader.")
        except Exception as e:
            log_msg(f"Failed to eject plate: {e}")
            raise

    def set_temp(self, temp: str):
        """
        Activate the plate reader's incubator and set it to a target temperature.
        Note that this command does not wait for the heating plates to reach the target temperature before proceeding.

        :return: None
        :raises: Exception if the set temp command fails.
        """
        try:
            self.exec(['Temp', temp])
            log_msg(f"Temperature set to {temp}.")
        except Exception as e:
            log_msg(f"Failed to set temperature: {e}")
            raise

    def run_protocol(self,
                     name: str,
                     test_path: str = r"C:\Users\Public\BMG\SPECTROstar Nano\User\Definit",
                     data_path: str = r"C:\Users\Public\BMG\SPECTROstar Nano\User\Data"
                     ):
        """
        Run a test protocol from pre-defined protocols stored on the plate reader.
        test_path and data_path variables should remain unchanged
            as these are default directories from BMG software install.

        :return: None
        :raises: Exception if the run protocol command fails.
        """
        try:
            # self.exec(['Run', name, test_path, data_path])
            self.com.ExecuteAndWait(['Run', name, test_path, data_path])
            log_msg(f"Protocol '{name}' completed successfully.")
        except Exception as e:
            log_msg(f"Failed to run protocol '{name}': {e}")
            raise

    def exec(self, cmd: list):
        """
        Eject the plate holder from the reader.

        :return: None
        :raises: Exception if the execute command fails.
        """
        try:
            res = self.com.ExecuteAndWait(cmd)
            if res:
                raise Exception(f"Command {cmd} failed: {res}")
        except Exception as e:
            log_msg(f"Command execution failed: {e}")
            raise


def get_most_recent_csv(directory: str):
    """
    Find the most recently modified CSV file in the specified directory.

    This function searches for all CSV files in the provided directory and returns the one with the latest modification time.

    :param directory: The directory path to search for CSV files.
    :raises FileNotFoundError: If no CSV files are found in the directory.
    :return: The file path of the most recently modified CSV file.
    """
    # Search for all CSV files in the given directory
    csv_files = glob.glob(os.path.join(directory, '*.csv'))

    if not csv_files:
        raise FileNotFoundError("No CSV files found in the directory.")

    # Sort files by their last modified time in descending order
    latest_file = max(csv_files, key=os.path.getmtime)

    return latest_file


def get_csv():
    """
    Retrieve the most recent CSV file from a predefined directory.

    This function identifies and returns the most recently modified CSV file from a specific directory containing experimental data.

    :return: The file path of the most recent CSV file.
    """
    # Specify the directory where CSV files are saved
    data_directory = r"C:\Users\Public\UV_VIS_DATA"

    # Get the most recent CSV file
    recent_csv = get_most_recent_csv(data_directory)

    return recent_csv


def measurements(bmg, protocol_name: str = 'Empty Plate Reading'):
    """
    Run a measurement protocol on the BMG SPECTROstar Nano reader.

    This function manages the process of ejecting the plate, inserting it, and executing the provided measurement protocol
    using the plate reader. It logs instrument statuses before and after each action.

    :param bmg: An instance of the BmgCom class controlling the plate reader.
    :param protocol_name: The name of the protocol to be run (default is 'Empty Plate Reading').
    :return: None
    """
    # Check instrument status
    log_msg(f"Instrument Status: {bmg.status()}")

    # Eject the plate
    # bmg.plate_out()

    # Insert the plate
    bmg.plate_in()
    log_msg(f"Instrument Status: {bmg.status()}")

    # # Set the target temperature
    # target_temp = '25.0'
    # bmg.set_temp(target_temp)

    # Define protocol parameters
    # protocol_name = 'Empty Plate Reading'

    test_runs_path = r"C:\Users\Public\BMG\SPECTROstar Nano\User\Definit"
    data_output_path = r"C:\Users\Public\BMG\SPECTROstar Nano\User\Data"

    bmg.run_protocol(protocol_name, test_runs_path, data_output_path)

    # bmg.plate_out()


def send_message(sock, message_type: str, message_data: str = ""):
    """
    Send a message to the server with a specified message type.

    The message is encoded as a string that combines the message type and optional message data, separated by a pipe ('|') character.
    The message is then sent to the server through the provided socket.

    :param sock: The socket object used to communicate with the server.
    :param message_type: The type of the message being sent (e.g., 'REQUEST', 'UPDATE').
    :param message_data: Optional additional data to include with the message (default is an empty string).
    :return: None
    """
    message = f"{message_type}|{message_data}"
    sock.sendall(message.encode())


def receive_message(sock):
    """
    Receive a message from the server.

    This function waits to receive a message from the server via the provided socket. The received message is split into
    its type and data components, using the pipe ('|') character as a delimiter.

    :param sock: The socket object used to receive the message.
    :return: A tuple containing the message type and message data.
    """
    data = sock.recv(1024).decode()
    return data.split("|", 1)


def handle_server(bmg, s):
    """
    Handle communication with a server (64-bit script).

    This function manages the interaction between the BMG SPECTROstar Nano reader and the server. It listens for
    messages from the server and executes the appropriate actions, such as performing background readings,
    running protocols, and collecting sample data. Depending on the message type, it performs the necessary operations
    on the plate reader and sends back CSV data or other responses to the server.

    :param bmg: An instance of the BmgCom class controlling the SPECTROstar Nano reader.
    :param s: A socket object used to communicate with the server.
    :raises Exception: If there is a failure in communication or plate reading operations.
    :return: None
    """
    try:
        bmg.plate_out()
        while True:
            # Wait for a message from the server
            log_msg("Awaiting message from server")
            msg_type, msg_data = receive_message(s)

            if msg_type == "PLATE_BACKGROUND":
                log_msg("Plate background requested")
                bmg.plate_out()
                measurements(bmg, msg_data)
                bmg.plate_out()
                plate_bg = get_csv()
                send_message(s, "PLATE_BACKGROUND", plate_bg)

            if msg_type == "RUN_PROTOCOL":
                measurements(bmg, msg_data)
                csv_file = get_csv()
                send_message(s, "CSV_FILE", csv_file)

            if msg_type == "GET_TEMP":
                temp_string = f"{bmg.temp1()}, {bmg.temp2()}"
                send_message(s, "TEMPS", temp_string)

            if msg_type == "SET_TEMP":
                bmg.set_temp(msg_data)
                send_message(s, "OK")

            if msg_type == "NEXT_READING":
                log_msg("Next sample requested.")
                log_msg("Ejecting reader & awaiting sample loading.")
                measurements(bmg)

            if msg_type == "SHUTDOWN":
                log_msg("Received signal from the server to shut down client.")
                break

            else:
                pass

        log_msg("Finished communication with 64-bit script.")

    except Exception as e:
        log_msg(f"Failed to communicate with 64-bit script: {e}")


def client_main():
    """
    Main function to establish communication between the BMG SPECTROstar Nano reader and the 64-bit server.

    This function initializes the connection to the server and handles the communication protocol by dispatching
    the ActiveX object for the plate reader and sending/receiving messages to/from the server. It handles the full
    lifecycle of the client-server interaction and ensures proper connection termination.

    :raises Exception: If an error occurs during communication or setup.
    :return: None
    """
    try:
        # Dispatch the ActiveX object
        bmg = BmgCom("SPECTROstar Nano")

        # Get software version
        bmg.version()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('localhost', 65432))  # Connect to the 64-bit script's server
            log_msg("Connected to server.")

            handle_server(bmg, s)

            log_msg("Disconnecting...")
            s.close()
            log_msg("Disconnected.")

    except Exception as e:
        log_msg(f"An error occurred: {e}")


if __name__ == '__main__':
    client_main()