"""Project-scoped client for the user's existing rhinomcp TCP bridge."""
import argparse,json,os,socket,sys
from pathlib import Path

def receive_exact(sock,n):
    out=bytearray()
    while len(out)<n:
        chunk=sock.recv(n-len(out))
        if not chunk:raise ConnectionError('Rhino bridge closed before complete response')
        out.extend(chunk)
    return bytes(out)

HOST=os.environ.get('RHINO_BRIDGE_HOST','127.0.0.1');PORT=int(os.environ.get('RHINO_BRIDGE_PORT','1999'))

def send(kind,params=None,timeout=45):
    body=json.dumps({'type':kind,'params':params or {}}).encode('utf-8')
    with socket.create_connection((HOST,PORT),timeout=timeout) as sock:
        sock.settimeout(timeout);sock.sendall(len(body).to_bytes(4,'big')+body)
        size=int.from_bytes(receive_exact(sock,4),'big')
        if size>64*1024*1024:raise ValueError('Unexpected Rhino response size')
        return json.loads(receive_exact(sock,size).decode('utf-8'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--cmd');p.add_argument('--script');p.add_argument('--output');p.add_argument('--timeout',type=int,default=45);a=p.parse_args()
    if bool(a.cmd)==bool(a.script):p.error('Choose either --cmd or --script')
    if a.script:
        code=Path(a.script).read_text(encoding='utf-8');result=send('execute_rhinoscript_python_code',{'code':code},a.timeout)
    else:result=send(a.cmd,timeout=a.timeout)
    if a.output:
        out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2)[:12000])
    details=result.get('result',{})
    if result.get('status')!='success' or isinstance(details,dict) and details.get('success') is False:sys.exit(1)
if __name__=='__main__':main()
