import http.server
import socketserver

PORT = 8765

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = '/sisyphus_cooling_web_interface.html'
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

Handler = MyHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at port {PORT}")
    print(f"Open http://localhost:{PORT} to view the Sisyphus Cooling Simulation")
    httpd.serve_forever()
