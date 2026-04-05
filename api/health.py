from server import app as root_app


def app(environ, start_response):
    scoped_environ = dict(environ)
    scoped_environ["PATH_INFO"] = "/health"
    return root_app(scoped_environ, start_response)
