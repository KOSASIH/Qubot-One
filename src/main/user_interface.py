from flask import render_template

class UserInterface:
    def render_home(self):
        return render_template('home.html')

    def render_status(self, status):
        return render_template('status.html ', status=status)

    def render_error(self, error_message):
        return render_template('error.html', error=error_message)
