About this fork
---------------

This fork lets you run Redis Server as a Windows service.

It introduces the redis-service.exe binary; this is a Windows service that
simply starts and stops the redis-server.exe binary.

It expects to find redis-server.exe and conf\redis.conf files in the same
directory as redis-service.exe.

This fork also introduces a Windows installer; this will:

  * install redis into Program Files.
  * configure redis to store its database dumps inside the "data"
    subdirectory, and logs inside "logs" subdirectory.
  * create the RedisService Windows account (with Logon as service
    privilege)
  * install a Windows Service to automatically start Redis Server (run
    as the RedisService account) at boot (but has to be manually
    started after install...).
  * the installer will not override existing redis.conf file (it does
    install a redis-dist.conf with the default settings).
  * create a bunch of Start Menu entries (for redis-cli.exe, redis home
    page, etc).

NB This is a work in progress. See the planned features inside the 
   src/service.c and setup/redis.iss files.

Let me known how this works for you!

  -- Rui Lopes


Windows 32 and x64 port of Redis server, client and utils
--------------------------------------------

It is made to be as close as possible to original unix version.
You can download prebuilt binaries here: 

   http://github.com/dmajkic/redis/downloads

Building Redis on Windows
-------------------------

Building Redis on Windows requires MinGW. If you are using full
mSysGit, you allready have all tools needed for the job. 

Start Git bash, and clone this repository:

   $ git clone http://github.com/dmajkic/redis.git

Compile it:

   $ make 

Test it: 

   $ make test 

Compiled programs are in source dir, and have no external dependencies.

You can use your own MinGW installation, RubyInstaller DevKit, or TDM. 
Note that you will need Tcl installed for testing. 

  
What is done and what is missing
--------------------------------

Commands that use fork() to perform backgroud operations are implemented 
as foreground operations. These are BGSAVE and BGREWRITEAOF. 
Both still work - only in foreground. All original tests pass.

Everything else is ported: redis-cli, hiredis with linenoise, rdb dumps, 
virtual memory with threads and pipes, replication, all commands, etc.

You can install and use all ruby gems that use Redis on windows.
You can develop on windows with local, native Redis server.
You can use redis-cli.exe to connect to unix servers.
...

Windows x64 port notice
-----------------------

Since there are more diferences between Linux and Windows 64bit systems,
and even if all tests suplied with redis pass, this port should be 
treated as experimental build, in need for more testing. 

To build it yourself you will need x64 gcc compiler (TDM or like).
Build procedure is same as 32 bit version. 

On 64bit windows, you can start 64bit redis-server from 32bit app
and use it to access more than 3,5Gb memory. 

Future plans
------------ 

Run tests, fix bugs, try to follow what Salvatore and Pieter are coding.

This port is bare. Redis-server.exe is console application, that can
be started from console or your app. It is not true Windows Service 
app, so there is space to make it SCM aware. 

That's it. Enjoy. 

Regads,
Dusan Majkic


Original redis README follows:
=============================================================================

Where to find complete Redis documentation?
-------------------------------------------

This README is just a fast "quick start" document. You can find more detailed
documentation at http://redis.io

Building Redis
--------------

It is as simple as:

    % make

You can run a 32 bit Redis binary using:

    % make 32bit

After building Redis is a good idea to test it, using:

    % make test

NOTE: if after building Redis with a 32 bit target you need to rebuild it
      with a 64 bit target you need to perform a "make clean" in the root
      directory of the Redis distribution.

Allocator
---------

By default Redis compiles and links against jemalloc under Linux, since
glibc malloc() has memory fragmentation problems.

To force a libc malloc() build use:

    % make FORCE_LIBC_MALLOC=yes

In all the other non Linux systems the libc malloc() is used by default.

On Mac OS X you can force a jemalloc based build using the following:

    % make USE_JEMALLOC=yes

Verbose build
-------------

Redis will build with a user friendly colorized output by default.
If you want to see a more verbose output use the following:

    % make V=1

Running Redis
-------------

To run Redis with the default configuration just type:

    % cd src
    % ./redis-server
    
If you want to provide your redis.conf, you have to run it using an additional
parameter (the path of the configuration file):

    % cd src
    % ./redis-server /path/to/redis.conf

Playing with Redis
------------------

You can use redis-cli to play with Redis. Start a redis-server instance,
then in another terminal try the following:

    % cd src
    % ./redis-cli
    redis> ping
    PONG
    redis> set foo bar
    OK
    redis> get foo
    "bar"
    redis> incr mycounter
    (integer) 1
    redis> incr mycounter
    (integer) 2
    redis> 

You can find the list of all the available commands here:

    http://redis.io/commands

Installing Redis
-----------------

In order to install Redis binaries into /usr/local/bin just use:

    % make install

You can use "make PREFIX=/some/other/directory install" if you wish to use a
different destination.

Make install will just install binaries in your system, but will not configure
init scripts and configuration files in the appropriate place. This is not
needed if you want just to play a bit with Redis, but if you are installing
it the proper way for a production system, we have a script doing this
for Ubuntu and Debian systems:

    % cd utils
    % ./install_server

The script will ask you a few questions and will setup everything you need
to run Redis properly as a background daemon that will start again on
system reboots.

You'll be able to stop and start Redis using the script named
/etc/init.d/redis_<portnumber>, for instance /etc/init.d/redis_6379.

Enjoy!
