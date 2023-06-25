import tkinter as tk
import ttkbootstrap

from ttkthemes import ThemedTk
from tkinter import PhotoImage, Label, ttk
from tkinter import font
from tkinter import Button


# style = ttk.Style()
# print(style.theme_names())
# style.theme_use('vista')  # Use the 'default' theme


class PaddedFrame(ttk.Frame):
    def __init__(self, parent, padding=10, **kwargs):
        super().__init__(parent, **kwargs)
        self.grid(padx=padding, pady=padding)

        # Load the logo
        self.logo = PhotoImage(file="logo.png")

        # Resize the logo to half its original size
        self.resized_logo = self.logo.subsample(2, 2)

        # Create a label to hold the logo
        self.logo_label = ttk.Label(self, image=self.resized_logo)

        # It ensures that the image is not garbage collected.
        self.logo_label.image = self.resized_logo

        # Position the logo in the top right corner of the frame
        self.logo_label.grid(row=0, column=1, sticky='ne')


class BasePage(ttk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid(sticky='nsew')
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        # Load the logo
        self.logo = PhotoImage(file="logo.png")

        # Create a label to hold the logo
        self.logo_label = Label(self, image=self.logo)

        # This line is important. Do not remove.
        # It ensures that the image is not garbage collected.
        self.logo_label.image = self.logo

        # Position the logo in the top right corner of the frame
        self.logo_label.grid(row=0, column=1, sticky='ne')


class RoundedButton(Button):
    def __init__(self, master=None, **kwargs):
        # Load the rounded button image
        self.rounded_button_image = PhotoImage(file="rb.png")

        # Set the image as the button's background
        kwargs['image'] = self.rounded_button_image
        kwargs['borderwidth'] = 0

        super().__init__(master, **kwargs)

        # This line is important. Do not remove.
        # It ensures that the image is not garbage collected.
        self.image = self.rounded_button_image


class MainApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("ADHD Screening Tool")
        self.style = ttkbootstrap.Style(theme='cerculean')
        self.option_add("*Font", "TkTooltipFont")
        fonts = list(font.families())
        fonts.sort()
        print("fonts = ", font.names(self))
        for f in fonts:
            print(f)
        # Import LoginPage within this method to avoid circular imports
        from Gui.LoginPage import loginPage
        print(font.names())
        self.frame_stack = []  # Stack to hold the frames
        self.frame = loginPage(self, self.switch_frame)
        self.frame.grid(sticky="nsew")  # use grid instead of pack
        self.grid_rowconfigure(0, weight=1)  # use grid instead of pack
        self.grid_columnconfigure(0, weight=1)  # use grid instead of pack

        # Create a new style that inherits from the TFrame style
        self.style.configure('White.TFrame', background='white')
        # Apply the new style to the frame
        self.frame['style'] = 'White.TFrame'

        # Configure the size of all buttons
        self.style.configure('TButton', width=20, height=5)  # Set the width and height

        # Create a new style for buttons with rounded corners
        # self.style.layout('Rounded.TButton',
        #                   [('Button.border',
        #                     {'children': [('Button.padding', {'children': [('Button.label', {'sticky': 'nswe'})],
        #                                                       'sticky': 'nswe'})],
        #                      'border': '10'})])  # '10' is the radius of the corners
        # self.style.configure('Rounded.TButton', font=('Helvetica', 12, 'bold'))
        self.attributes("-fullscreen", False)
        self.resizable(False, False)

    def switch_frame(self, frame_class=None, model=None, **kwargs):
        """Hides current frame and replaces it with a new one."""
        # Push the current frame to the stack
        if self.frame is not None:
            self.frame.grid_remove()  # Hide the current frame
            self.frame_stack.append(self.frame)
        self.model = model  # Update the model
        # Remove the 'callback' key from kwargs if it exists
        kwargs.pop('callback', None)
        # Create a new frame instance and set it as the current frame
        self.frame = frame_class(self, callback=self.go_back, **kwargs)

        # Replace all buttons in the frame with rounded buttons
        for child in self.frame.winfo_children():
            if isinstance(child, ttk.Button):
                options = child.config()

                # Check if options is not None
                if options is not None:
                    # Create a new rounded button with the same options
                    rounded_button = RoundedButton(self.frame, **options)

                    # Place the rounded button in the same grid cell as the original button
                    rounded_button.grid(row=child.grid_info()['row'], column=child.grid_info()['column'])

                    # Destroy the original button
                    child.destroy()

        self.frame.grid(sticky="nsew")  # use grid instead of pack

    def update_model(self, model):
        self.model = model  # Update the model
        # Switch back to the current frame, which will now receive the updated model
        self.switch_frame(self.frame.__class__, model=self.model)

    def close(self):
        self.destroy()

    def go_back(self):
        """Go back to the previous frame."""
        if self.frame_stack:
            # Pop the last frame from the stack and destroy the current frame
            last_frame = self.frame_stack.pop()
            self.frame.grid_remove()  # Hide the current frame
            # Set the last frame as the current frame
            self.frame = last_frame
            self.frame.grid(sticky="nsew")  # use grid instead of pack


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
