# -*- coding: utf-8 -*-

class Logger:
    def __init__(self, path):
        self.path = path
        self.writer = open(self.path, 'w', encoding='utf-8')

    def __call__(self, s):
        print(s, flush=True)
        self.writer.write(s+'\n')

    def __del__(self):
        self.writer.close()
