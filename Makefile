flags= -lm -Wall -Wextra -Werror -pedantic


all:
	gcc src/*.c -o build/xeuron $(flags)

clean:
	rm -rf build/* && clear