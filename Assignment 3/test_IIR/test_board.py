import pyfirmata2

# Let pyFirmata2 auto-detect the port and connect
board = pyfirmata2.Arduino(pyfirmata2.Arduino.AUTODETECT)

print("Board connected!")

board.exit()
