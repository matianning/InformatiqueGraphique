#!/bin/sh
bindir=$(pwd)
cd /home/tianning/Bureau/TP3_lissage/TP3_lissage/
export 

if test "x$1" = "x--debugger"; then
	shift
	if test "xYES" = "xYES"; then
		echo "r  " > $bindir/gdbscript
		echo "bt" >> $bindir/gdbscript
		/usr/bin/gdb -batch -command=$bindir/gdbscript --return-child-result /home/tianning/Bureau/TP3_lissage/build/TP3_lissage 
	else
		"/home/tianning/Bureau/TP3_lissage/build/TP3_lissage"  
	fi
else
	"/home/tianning/Bureau/TP3_lissage/build/TP3_lissage"  
fi
