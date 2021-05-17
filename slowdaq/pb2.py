from __future__ import print_function
from . netstring import *
from . netarray import *
from . stream import *
from . snapshot import *
import os
import json
import datetime
import time

try:
    # only needed for aggregator and log related functionality
    import pandas as pd
    from . logreader import *
except:
    pass

defaultRunID = 'Run22300000'
defaultsubRunID = '000'

class Publisher(Server):
    """
    Sends data to an Aggregator. Also capable of recieving messages from
    Aggregators and Subscribers.
    """
    def __init__(self,name, agg_addr, agg_port,debug=False):

        Server.__init__(self,debug)

        self.name = name
        self.agg_addr = agg_addr
        self.agg_port = agg_port
        self.pid = os.getpid()

        # for display purposes only
        self.status = 'unset'
        self.status_color = 'white'

        self.aggregator = self.add_connection(agg_addr,agg_port)

        # listen on any open port
        self.add_listener()

        # all messages sent to this instance end up here
        self.inbox = deque([],maxlen=128)

        self.pulse()

    def pack(self,d):
        """
        Add key-value pairs to a dict d so as to be intelligible to an
        aggregator recieving it over the network, and return it as sendable JSON
        """
        d['source'] = [self.name, self.pid]
        d['event'] = 'data'
        apply_timestamp(d)
        return json.dumps(d)

    def pulse(self):
        """
        queue a heartbeat signal to send to all clients
        """

        p = Pulse(self)

        self.queue(p.serialize())
        self.t_last_pulse = now()

    def handle_recv(self,stream,msgs):
        self.inbox += msgs

    def serve(self,timeout=0):
        # decide whether or not to create and publish a heartbeat
        t = now()
        if t - self.t_last_pulse > datetime.timedelta(seconds=15):
            self.pulse()
        Server.serve(self,timeout)

    def request_newfile(self, postfix = ''):
        d = {'event': 'request_newfile'}
        d['postfix'] = postfix
        self.aggregator.queue(json.dumps(d))

class Aggregator(Server):
    """
    Broker for a basic slowdaq system. Collects data from Publishers, and
    informs Subscribers of how to connect to any active Publishers by itself
    publishing a snapshot of the slowdaq system state.
    """
    def __init__(self,addr,port,logdir,debug=False):

        Server.__init__(self,debug)

        self.add_listener(addr,port)

        self.data = []

        # setup an initial snapshot
        self.snapshot = Snapshot()

        self.pandas_buffer = {}
        self.pandas_lines = 0
        self.pandas_source_list = []

        self.logdir = logdir

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        self.logfname = ''
        self.logfname_post = ''
        self.logfname_postadd = '_' + defaultRunID + '_' + defaultsubRunID
        self.create_newlogf = False

    def process_single_data_dict(self,d):
        """
        Insert d into the appropriate pandas dataframe prior to
        dataframe logging
        """
        name,parsed = process_line(flatten(d))
        if not name in self.pandas_source_list: return
        if name in self.pandas_buffer:
            self.pandas_buffer[name].append(parsed)
        else:
            self.pandas_buffer[name] = [parsed]

        self.pandas_lines += 1

    def emit_pandas_log(self):
        """
        Note: clears pandas_buffer
        """
        for name in self.pandas_buffer:
            fname = "%s_%s.pkl" % (name, self.logfname_post)
            fname = os.path.join(self.logdir,fname)
            df = pd.DataFrame(self.pandas_buffer[name])
            df.to_pickle(fname)

        self.pandas_buffer = {}
        self.pandas_lines = 0

    def log(self):
        """
        Note: clears self.data
        """

        if not os.path.exists(self.logfname):
            self.logfname_post = timestamp(utcnow(),timestamp_fmt_log) + self.logfname_postadd
            fname = "slowdaq_%s.log" % (self.logfname_post)
            fname = os.path.join(self.logdir, fname)
            self.logfname = fname
            # generating symlink for the recent log file
            fname_symlink = "slowdaq_most_recent.log"
            try:
                os.remove(fname_symlink)
            except OSError:
                pass
            os.symlink(self.logfname, fname_symlink)

        # Starting a new file
        if os.path.exists(self.logfname) and (os.path.getsize(self.logfname) > 500e6) or self.create_newlogf:
            self.emit_pandas_log()
            # change file name
            self.logfname_post = timestamp(utcnow(),timestamp_fmt_log) + self.logfname_postadd
            fname = "slowdaq_%s.log" % self.logfname_post
            fname = os.path.join(self.logdir, fname)
            self.logfname = fname

            # generating symlink for the recent log file
            fname_symlink = "slowdaq_most_recent.log"
            try:
                os.remove(fname_symlink)
            except OSError:
                pass
            os.symlink(self.logfname, fname_symlink)
            self.create_newlogf = False

        # Add header info when we create a new file
        if not os.path.exists(self.logfname):
            d = {}
            d['event'] = 'header'
            d['source'] = 'header'
            d['filename'] = self.logfname
            d['time'] = time.time()
            d['id'] = self.logfname_postadd
            apply_timestamp(d)
            self.data.insert(0, json.dumps(d))

        with File(self.logfname,'a') as f:
            while len(self.data) > 0:
                msg = self.data.pop(0)
                f.write(msg)
            f.flush()            

    def handle_recv(self,stream,msgs):

        do_new_snapshot = False

        for msg in msgs:
            try:
                m = json.loads(msg)
                if m == None: continue
                if not isinstance(m,dict): continue
            except:
                continue

            if m['event'] == 'emit_log':
                self.emit_pandas_log()
                self.log()

            if m['event'] == 'data':
                self.data.append(msg)
                if len(self.pandas_source_list):
                    self.process_single_data_dict(m)

            if m['event'] == 'request_snapshot':
                # defer working on this at all until after sifting through all
                # the messages we have for new pulse events
                do_new_snapshot = True

            if m['event'] == 'pulse':
                # older pulses from the same source are deleted
                addr,port = stream.remote_location
                # Recall that the pulse contains only information that the
                # the publisher has, and that the publisher alone knows
                # the port it's listening on, but doesn't know the address
                p = Pulse(m)
                self.snapshot.update(p,addr)

            if m['event'] == 'request_newfile':
                print('Received a message for new file', m)
                self.create_newlogf = True
                self.logfname_postadd = m['postfix']

        # snapshot logging period needs to be faster than three minutes since
        # we purge anything older than that from it
        self.snapshot.purge_stale(datetime.timedelta(minutes=3))
        self.live = self.snapshot.locations

        if do_new_snapshot:
            stream.queue(self.snapshot.serialize())

    def handle_accept(self,stream):
        stream.queue(self.snapshot.serialize())

class Subscriber(Server):
    """
    Handles the automatic retrieval of system snapshots from an Aggregator, and
    subscribes to any data published by the Publishers connected to that
    Aggregator.

    It is not recommended that a Subscriber be used as a data archiver.
    Subscribers are more prone to data loss than Aggregators and Archivers
    because they are one level further removed from the Publishers. When a
    Publisher's connection to a Subscriber fails, the Publisher has no
    responsibility to buffer for the Subscriber any data generated until the
    connection is reestablished. Further, the Subscriber must reestablish that
    connection itself by taking potentially seconds to get a new snapshot from
    the Aggregator.
    """

    def __init__(self, agg_addr, agg_port,debug=False):

        Server.__init__(self,debug)

        self.agg_addr = agg_addr
        self.agg_port = agg_port

        self.aggregator = self.add_connection(agg_addr,agg_port)

        # Will be filled with all data frames recieved. Up to the user to empty.
        self.data = deque([],maxlen=512)

        self.snapshot = None

    def request_snapshot(self):
        d = {'event':'request_snapshot'}
        self.aggregator.queue(json.dumps(d))

    def handle_close(self,stream):
        # if we lose a connection, it's safest to ask the aggregator if it's
        # still there before attempting to reconnect
        self.request_snapshot()
        Server.handle_close(self,stream)

    def register_snapshot(self,snapshot):
        """
        Set the live socket dictionary to a snapshot and make sure all of its
        entries are connected.
        """

        self.snapshot = snapshot

        # removal of nonlive connections will be done for us during self.serve
        self.live = self.snapshot.locations

        for addr,port in self.snapshot.locations:
            # if already connected, silently does nothing

            stream = self.add_connection(addr,port)
            self.snapshot.locations[(addr,port)].stream = stream

    def handle_recv(self,stream,msgs):

        for msg in msgs:

            try:
                m = json.loads(msg)
            except:
                continue

            if m['event'] == 'snapshot':

                self.register_snapshot(Snapshot(m))


            if m['event'] == 'data':
                self.data.append(m)


    def serve(self,timeout=0):
        self.request_snapshot()
        Server.serve(self,timeout)


    def message(self,name,msg):

        if self.snapshot == None:
            return False

        if name in self.snapshot.names:
            if isinstance(msg,dict):
                msg = json.loads(msg)

            try:
                self.snapshot.names[name].stream.queue(msg)
                return False
            except:
                return False
        else:
            return False


class Archiver(Server):
    pass
