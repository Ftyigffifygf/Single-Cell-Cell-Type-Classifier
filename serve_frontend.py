import http.server
import socketserver
import os
import webbrowser
from threading import Timer

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def open_browser():
    webbrowser.open('http://localhost:3000')

if __name__ == '__main__':
    PORT = 3000
    
    # Change to frontend directory
    os.chdir('frontend')
    
    # Start server
    with socketserver.TCPServer(('', PORT), CORSHTTPRequestHandler) as httpd:
        print(f'Frontend server running at http://localhost:{PORT}')
        print('Make sure the API server is running on http://localhost:8000')
        
        # Open browser after 1 second
        Timer(1.0, open_browser).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\\nShutting down frontend server...')
            httpd.shutdown()
