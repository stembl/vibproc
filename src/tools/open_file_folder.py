## Returns a path name to a file or folder
import tkinter as tk
from tkinter import filedialog

def get_path():
    '''
    Pops up a small GUI allowing user to select either a
    file or folder containing the data.
    '''
    class Application(object):
        def __init__(self):
            self.value = None
            self.root = None

        def get_path2(self):
            '''
            Show user window, and wait for the user to click a button.
            Allows user to search for a File or Folder depending on
            button selected.
            '''

            self.root = tk.Tk()
            self.root.lift
            self.root.title('File Selection')
            self.root.geometry("200x80")

            ftext = tk.Label(text='Open a File or Folder?')
            ftext.pack()

            file_button = tk.Button(self.root, text = "File",
                                    command= lambda: self.finish('file'))
            folder_button = tk.Button(self.root, text = "Folder",
                                     command= lambda: self.finish('folder'))

            file_button.pack()
            folder_button.pack()

            # start the loop, and wait for the dialog to be
            # destroyed. Then, return the value:
            self.root.mainloop()
            return self.value

        def finish(self, ftype):
            '''
            Set the value and close the window
            This will cause the get_path() function to return.
            '''

            if ftype=='file':
                file_path = filedialog.askopenfilename(filetypes=[("Two Column CSV","*.csv")])
            if ftype=='folder':
                file_path = filedialog.askdirectory()
            self.value = file_path
            self.root.destroy()

    path = Application().get_path2()

    return(path)
