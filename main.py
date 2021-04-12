import heapq
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import math
import logging

logging.getLogger().setLevel(logging.INFO)

N_INTERESTS = 20
N = 1000
INIT_NUM_NODES = 30
NUM_OF_PEOPLE_EACH_SMALL_NETWORK_NODE = 2

def sample_exponential(parameter):
    value = np.random.exponential(parameter, 1)
    return value


def logistic_function(x):
    L = 1  # Curve's max value
    k = 1  # Controlling the growth rate
    x0 = 10 # x = 10 yields value = 0.5
    value = L/(1+np.exp(-k*(-x+x0)))
    return value


class EventQueue(object):
    def __init__(self):
        self.queue = []

    def add_event(self, event):
        heapq.heappush(self.queue, event)

    def step(self, t):
        while self.queue and self.queue[0].t <= t: # if closest event time is less than current time
            closest_event = heapq.heappop(self.queue)
            closest_event.callback()


class Event(object):
    def __init__(self, trigger_time):
        self.t = trigger_time

    def __lt__(self, other):
        return self.t < other.t

    def __eq__(self, other):
        return self.t == other.t

    def callback(self):
        raise NotImplementedError()



class PropagationEvent(Event):
    def __init__(self, current_time, n1, n2, event_queue): #The propagation occurs from n1->n2
        self.n1 = n1
        self.n2 = n2
        self.current_time = current_time
        self.event_queue = event_queue

        self.exp_parameter = 5
        trigger_time = sample_exponential(self.exp_parameter) + self.current_time
        super().__init__(trigger_time)
        logging.debug(f"PropagationEvent created trigger time: {trigger_time}")

    def callback(self):
        if not self.n2.bought_the_product: # If n2 still has not purchased the product by the time callback is triggered
            h = self.n1.neighb_strength[self.n2] # Edge Strength
            q = np.random.uniform(low=0, high=1) #  Probability of Propagation
            if q <= h:
                logging.debug(f" Propagation Event successfully bought edge strength: {h} trigger time: {self.t}")

                # propagation occurs and schedule new propagation events
                self.n2.bought_the_product = True
                for neighbor in self.n2.neighbours:
                    if not neighbor.bought_the_product: # If the neighbor still has not purchased the product by the time callback is scheduled
                        self.event_queue.add_event(PropagationEvent(math.ceil(self.t), self.n2, neighbor, self.event_queue))

    def __repr__(self):
        return f" PropagationEvent t: {self.t}"

class UpdateEdgeEvent(Event):
    def __init__(self, current_time, n1, n2, event_queue):
        self.exp_parameter = 5
        trigger_time = sample_exponential(self.exp_parameter) + current_time
        super().__init__(trigger_time)

        self.event_queue = event_queue
        self.n1 = n1
        self.n2 = n2
        logging.debug(f"update edge event created trigger time: {trigger_time}")


    def callback(self):
        H = np.inner(self.n1.interests, self.n2.interests)/len(self.n1.interests) * \
            logistic_function(len(self.n1.neighb_strength.keys())) * logistic_function(len(self.n2.neighb_strength.keys()))

        logging.debug(f"update edge event triggered H : {H}, trigger time {self.t}")
        assert H <= 1 and H >= 0


        if H > 0.2:
            # form or update edge to be H
            self.n1.neighb_strength[self.n2] = H
            self.n2.neighb_strength[self.n1] = H

        elif H > 0.1:
            # if there is already an edge, update, if not, do nothing since not strong enough to form new edge
            if self.n2 in self.n1.neighb_strength:

                assert self.n1 in self.n2.neighb_strength

                self.n1.neighb_strength[self.n2] = H

            if self.n1 in self.n2.neighb_strength:

                assert self.n2 in self.n1.neighb_strength

                self.n2.neighb_strength[self.n1] = H

        else: # H <= 0.1
            # if edge exist, remove it otherwise do nothing
            if self.n2 in self.n1.neighb_strength:
                del self.n1.neighb_strength[self.n2]

            if self.n1 in self.n2.neighb_strength:
                del self.n2.neighb_strength[self.n1]

        self.event_queue.add_event(UpdateEdgeEvent(math.ceil(self.t), self.n1, self.n2, self.event_queue))

    def __repr__(self):
        return f" UpdateEdgeEvent t: {self.t}"


class SmallNetwork(object):
    def __init__(self, start_time, event_queue, init_num_nodes=INIT_NUM_NODES): # initially how many nodes are in the network
        self.nodes = [Node() for i in range(init_num_nodes)]
        heapq.heapify(self.nodes)

        self.add_all_edge_update_event(event_queue, start_time)
        self.event_queue = event_queue

    def add_all_edge_update_event(self, event_queue, start_time):
        for i in range(len(self.nodes)):
            for j in range(i+1, len(self.nodes)):
                event_queue.add_event(UpdateEdgeEvent(start_time, self.nodes[i], self.nodes[j], event_queue))

    def add_node(self, current_time):
        new_node = Node()
        for node in self.nodes:
            self.event_queue.add_event(UpdateEdgeEvent(current_time, new_node, node, self.event_queue))
        heapq.heappush(self.nodes, new_node)
        logging.info(f"add node, current number of nodes {len(self.nodes)}")

    def delete_node(self): # remove the nodes with the lowest activeness

        if len(self.nodes) > 0:
            remove_node = heapq.heappop(self.nodes)
            remove_node.remove_all_neighb_edges_with_me()
            logging.info(f"delete node, current number of nodes {len(self.nodes)}")
            return

        logging.debug(f"delete node, current number of nodes {len(self.nodes)}, cannot delete")

    def get_total_number_people_buys(self):
        result = 0
        for node in self.nodes:
            if node.bought_the_product:
                result += 1

        return result

    def bfs_and_get_total_strength(self, node):
        max_level = 5
        queue = deque([(1, node, 0)])
        visited = set()

        total_strength = 0
        while len(queue) > 0:

            curr_level, curr, edge_strength_got_here = queue.popleft()
            if curr_level > max_level:
                break

            total_strength += edge_strength_got_here
            visited.add(curr)

            for neigb in curr.neighbours:
                if neigb not in visited:
                    queue.append((curr_level + 1, neigb, curr.neighb_strength[neigb]))

        return total_strength




    def find_best_node_for_ad(self):
        #TODO currently it's a heuristic, formulate it as a maximization problem and develop an algorithm for it
        best_node_total_strength = -float("inf")
        best_node = None
        for node in self.nodes:
            total_strength = self.bfs_and_get_total_strength(node)
            if total_strength > best_node_total_strength:
                best_node_total_strength = total_strength
                best_node = node

        return best_node, best_node_total_strength

    def get_average_activeness(self):
        activenesses = [node.activeness for node in self.nodes]
        return np.mean(activenesses)


class BigNetwork(object):
    def __init__(self):
        initial_big_network_size = INIT_NUM_NODES * NUM_OF_PEOPLE_EACH_SMALL_NETWORK_NODE
        self.S = N - initial_big_network_size
        self.I = initial_big_network_size
        self.S_prev = None
        self.I_prev = None
        self.S_history = []
        self.I_history = []

    def update_small_network(self, small_network, current_time):
        last_step_difference=self.I - self.I_prev
        logging.debug(f"last_step_difference {last_step_difference}")

        if last_step_difference > NUM_OF_PEOPLE_EACH_SMALL_NETWORK_NODE:
            for i in range(int(last_step_difference//NUM_OF_PEOPLE_EACH_SMALL_NETWORK_NODE)) :
                 small_network.add_node(current_time)
        elif last_step_difference < -NUM_OF_PEOPLE_EACH_SMALL_NETWORK_NODE:
            for i in range(int((-last_step_difference)//NUM_OF_PEOPLE_EACH_SMALL_NETWORK_NODE)) :
                small_network.delete_node()


    def step(self, small_network):

        self.S_history.append(self.S)
        self.I_history.append(self.I)

        self.S_prev = self.S
        self.I_prev = self.I

        beta = small_network.get_average_activeness()
        logging.info(f"beta: {beta}")
        if beta > 0.75:
            gamma = 0.1
        else:
            gamma = 0.1*(0.85 - beta)

        self.S = self.S_prev - beta * self.S_prev * self.I_prev / N + gamma * self.I_prev
        self.I = self.I_prev + beta * self.S_prev * self.I_prev / N - gamma * self.I_prev


    def SIplot(self):
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
        ax.plot(np.array(self.S_history) / N, 'b', alpha=0.5, lw=2, label='Potential User')
        ax.plot(np.array(self.I_history) / N, 'r', alpha=0.5, lw=2, label='Active')
        ax.set_xlabel('Time /days')
        ax.set_ylabel('Number (1000s)')
        ax.set_ylim(0, 1.2)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(b=True, which='major', c='w', lw=2, ls='-')
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)
        plt.savefig("bignetwork SIplot.jpg")


class Node(object):
    def __init__(self):
        self.neighb_strength = {}  # dict of nodes:edge_strength
        self.interests = self.init_interest(N_INTERESTS) # np array of [0,1]

        self.bought_the_product = False

    @property
    def activeness(self):
        if len(self.neighbours) == 0:
            return 0
        else:
            return sum(self.neighb_strength.values()) / len(self.neighbours)


    @property
    def neighbours(self):
        return self.neighb_strength.keys()

    def init_interest(self, n_interests):
        # uniformly sample N_INTERESTS interests
        return np.random.uniform(low=0, high=1, size=n_interests)

    def __lt__(self, other):
        return self.activeness < other.activeness

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))

    def remove_all_neighb_edges_with_me(self): # remove all neighb edge's with me in neighb's object since im getting deleted anyways
        for neighb in self.neighbours:
            del neighb.neighb_strength[self]

    def place_advertisement(self, t, event_queue):
        # buy and add a propagation event
        self.bought_the_product = True
        for neighbor in self.neighb_strength.keys():
            event_queue.add_event(PropagationEvent(t, self, neighbor, event_queue))


def main():

    total_number_people_buys = []
    for random_seed in range(0,10):

        # Phase 1: Form the small network
        small_network_form_time = 200
        t = 0

        event_queue = EventQueue()
        big_network = BigNetwork()
        small_network = SmallNetwork(t, event_queue, init_num_nodes=INIT_NUM_NODES)

        # for visualization
        small_network_num_nodes_history = [len(small_network.nodes)]
        small_network_average_activeness_history = [small_network.get_average_activeness()]

        while t < small_network_form_time:
            logging.info(f"***********current time {t} ***************")
            big_network.step(small_network)
            logging.info(f"I(t) at t: {t} big_network I {big_network.I}")

            big_network.update_small_network(small_network, t)
            event_queue.step(t) # evolve the networks
            t += 1


            small_network_num_nodes_history.append(len(small_network.nodes))
            small_network_average_activeness_history.append(small_network.get_average_activeness())
        plt.figure(0)
        plt.plot(small_network_num_nodes_history)
        plt.savefig("small_network_num_nodes_history.jpg")

        plt.figure(1)
        plt.plot(small_network_average_activeness_history)
        plt.savefig("small_network_average_activeness_history.jpg")

        plt.figure(2)
        plt.plot(big_network.I_history)
        plt.savefig("big_network I_history.jpg")

        big_network.SIplot()
        # Phase 2: simulate advertisement propagation
        adver_time = 20 #days
        adver_start_node, _ = small_network.find_best_node_for_ad()

        adver_start_node.place_advertisement(t, event_queue)
        while t < adver_time + small_network_form_time:
            event_queue.step(t)
            t+=1
        total_number_people_buys.append(small_network.get_total_number_people_buys())

    logging.info( f"mean total_number_people_buys:  {np.mean(total_number_people_buys)}")

main()