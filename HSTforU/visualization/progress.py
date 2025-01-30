
# https://stackoverflow.com/questions/6169217/replace-console-output-in-python
def progress_bar(current, total, bar_length=20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * bar_length - 1) + '>'
    spaces  = ' ' * (bar_length - len(arrow))

    print('Progress: [%s%s] %d%% (%d/%d)' % (arrow, spaces, percent, current, total), end='\r')