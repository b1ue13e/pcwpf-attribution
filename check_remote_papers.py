import paramiko

def check_server(port):
    print(f"Connecting to 192.168.17.251:{port}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect("192.168.17.251", port=port, username="root", password="061022", timeout=10)
        cmd = "find /root -type f -name '*.pdf' 2>/dev/null; find /root -maxdepth 3 -type d 2>/dev/null"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode())
        ssh.close()
    except Exception as e:
        print(f"Error on port {port}: {e}")

if __name__ == "__main__":
    check_server(25329)
    check_server(25979)
