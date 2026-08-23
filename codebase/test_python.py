import socket

def check_environment():
    # Get the current server hostname
    hostname = socket.gethostname()
    
    # Define a test port number
    test_port = 8080
    
    # Print the results
    print(f"Server Hostname: {hostname}")
    print(f"Test Port: {test_port}")
    print("Python environment is working correctly.")

if __name__ == "__main__":
    check_environment()