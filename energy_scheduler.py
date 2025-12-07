from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import math
import matplotlib.pyplot as plt

# --------------------------------
# Data Models
# --------------------------------

@dataclass
class Process:
    pid: str
    arrival: int
    burst: int
    remaining: int = field(init=False)
    start_time: int = None
    completion_time: int = None
    waiting_time: int = None
    turnaround_time: int = None

    def __post_init__(self):
        self.remaining = self.burst


@dataclass
class ExecSegment:
    pid: str
    start: int
    end: int
    freq: float  # CPU frequency used during this segment


# ------------------------------
# Scheduler Base Class
# ------------------------------

class SchedulerBase:
    name: str = "BaseScheduler"

    def run(self, processes: List[Process]) -> Tuple[List[Process], List[ExecSegment]]:
        raise NotImplementedError("Subclasses must implement run()")


# ------------------------------
# FCFS Scheduler (Baseline)
# ------------------------------

class FCFSScheduler(SchedulerBase):
    name = "FCFS (High Frequency)"

    def __init__(self, freq: float = 2.0):
        self.freq = freq

    def run(self, processes: List[Process]) -> Tuple[List[Process], List[ExecSegment]]:
        procs = sorted(
            [Process(p.pid, p.arrival, p.burst) for p in processes],
            key=lambda p: (p.arrival, p.pid),
        )
        time = 0
        segments: List[ExecSegment] = []

        for p in procs:
            if time < p.arrival:
                time = p.arrival
            p.start_time = time
            start = time

            exec_time = p.burst
            time += exec_time

            p.completion_time = time
            p.turnaround_time = p.completion_time - p.arrival
            p.waiting_time = p.turnaround_time - p.burst

            segments.append(ExecSegment(p.pid, start, time, self.freq))

        return procs, segments


# ------------------------------
# Round Robin Scheduler
# ------------------------------

class RRScheduler(SchedulerBase):
    name = "Round Robin (High Frequency)"

    def __init__(self, quantum: int = 2, freq: float = 2.0):
        self.quantum = quantum
        self.freq = freq

    def run(self, processes: List[Process]) -> Tuple[List[Process], List[ExecSegment]]:
        procs = [Process(p.pid, p.arrival, p.burst) for p in processes]
        time = 0
        segments: List[ExecSegment] = []
        ready: List[Process] = []
        procs_sorted = sorted(procs, key=lambda p: (p.arrival, p.pid))

        i = 0
        completed = 0
        n = len(procs_sorted)

        while completed < n:
            while i < n and procs_sorted[i].arrival <= time:
                ready.append(procs_sorted[i])
                i += 1

            if not ready:
                time = procs_sorted[i].arrival
                continue

            current = ready.pop(0)

            if current.start_time is None:
                current.start_time = time

            run_time = min(self.quantum, current.remaining)
            start = time
            time += run_time
            current.remaining -= run_time

            segments.append(ExecSegment(current.pid, start, time, self.freq))

            while i < n and procs_sorted[i].arrival <= time:
                ready.append(procs_sorted[i])
                i += 1

            if current.remaining > 0:
                ready.append(current)
            else:
                current.completion_time = time
                current.turnaround_time = current.completion_time - current.arrival
                current.waiting_time = current.turnaround_time - current.burst
                completed += 1

        return procs_sorted, segments


# ------------------------------
# Energy-Aware Scheduler (DVFS)
# ------------------------------

class EnergyAwareScheduler(SchedulerBase):
    name = "Energy-Aware FCFS with DVFS"

    def __init__(
        self,
        high_freq: float = 2.0,
        mid_freq: float = 1.7,
        low_freq: float = 1.3,
        threshold_high: int = 10,
        threshold_mid: int = 5,
    ):
        self.high_freq = high_freq
        self.mid_freq = mid_freq
        self.low_freq = low_freq
        self.threshold_high = threshold_high
        self.threshold_mid = threshold_mid

    def run(self, processes: List[Process]) -> Tuple[List[Process], List[ExecSegment]]:
        procs = sorted(
            [Process(p.pid, p.arrival, p.burst) for p in processes],
            key=lambda p: (p.arrival, p.pid),
        )
        time = 0
        segments: List[ExecSegment] = []
        base_freq = self.high_freq

        for idx, p in enumerate(procs):
            if time < p.arrival:
                time = p.arrival

            total_remaining = sum(x.burst for x in procs[idx:])

            if total_remaining > self.threshold_high:
                freq = self.high_freq
            elif total_remaining > self.threshold_mid:
                freq = self.mid_freq
            else:
                freq = self.low_freq

            effective_speed = freq / base_freq
            exec_time = math.ceil(p.burst / effective_speed)

            p.start_time = time
            start = time
            time += exec_time
            p.completion_time = time
            p.turnaround_time = p.completion_time - p.arrival
            p.waiting_time = p.turnaround_time - p.burst

            segments.append(ExecSegment(p.pid, start, time, freq))

        return procs, segments


# ------------------------------
# Energy Model & Metrics
# ------------------------------

def power(freq: float) -> float:
    return freq ** 3


def compute_metrics(procs: List[Process], segments: List[ExecSegment]) -> Dict[str, float]:
    n = len(procs)
    avg_wait = sum(p.waiting_time for p in procs) / n
    avg_turn = sum(p.turnaround_time for p in procs) / n

    makespan = max(p.completion_time for p in procs) - min(p.arrival for p in procs)
    busy = sum(seg.end - seg.start for seg in segments)
    util = busy / makespan if makespan > 0 else 0.0

    energy = sum(power(seg.freq) * (seg.end - seg.start) for seg in segments)

    return {
        "avg_waiting_time": avg_wait,
        "avg_turnaround_time": avg_turn,
        "cpu_utilization": util,
        "total_energy": energy,
    }


# ------------------------------
# Pretty Printing
# ------------------------------

def print_process_table(processes: List[Process]) -> None:
    print("\nProcesses:")
    print("+-------+----------+--------+")
    print("| PID   | Arrival  | Burst  |")
    print("+-------+----------+--------+")
    for p in processes:
        print(f"| {p.pid:<5} | {p.arrival:^8} | {p.burst:^6} |")
    print("+-------+----------+--------+\n")


def print_metrics_table(metrics: Dict[str, float]) -> None:
    print("Metrics:")
    print("+----------------------+------------+")
    print(f"| {'Avg waiting time':<20} | {metrics['avg_waiting_time']:>8.2f} |")
    print(f"| {'Avg turnaround':<20}  | {metrics['avg_turnaround_time']:>8.2f} |")
    print(f"| {'CPU utilization':<20} | {metrics['cpu_utilization']*100:>7.2f}% |")
    print(f"| {'Total energy':<20}    | {metrics['total_energy']:>8.2f} |")
    print("+----------------------+------------+\n")


def print_gantt(segments: List[ExecSegment], title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not segments:
        print("(no segments)\n")
        return

    print("Each character ≈ 1 time unit.")
    print("Top line: which process ran.")
    print("Second line: frequency level (H=High, M=Medium, L=Low).\n")

    timeline = ""
    freq_line = ""
    time_marks = ""
    last_end = segments[0].start

    for seg in segments:
        if seg.start > last_end:
            gap = seg.start - last_end
            timeline += " " * gap
            freq_line += " " * gap
            for t in range(last_end, seg.start):
                time_marks += "|" if t % 5 == 0 else " "

        run_len = seg.end - seg.start
        block = seg.pid * run_len
        timeline += block

        if seg.freq >= 1.9:
            freq_char = "H"
        elif seg.freq >= 1.5:
            freq_char = "M"
        else:
            freq_char = "L"
        freq_line += freq_char * run_len

        for t in range(seg.start, seg.end):
            time_marks += "|" if t % 5 == 0 else " "

        last_end = seg.end

    print("Processes:")
    print(timeline)
    print("Freq lvl:")
    print(freq_line)
    print(time_marks)
    print("Legend: H=High freq, M=Medium, L=Low, '|' every 5 time units\n")


# ------------------------------
# Input Helper
# ------------------------------

def show_energy_bar_chart(results: List[Dict[str, float]]) -> None:
    """
    Show a bar chart comparing total energy for each algorithm.
    `results` is the list of dicts we build in run_experiment().
    """
    names = [r["name"] for r in results]
    energies = [r["energy"] for r in results]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(names, energies)

    plt.title("Energy Consumption Comparison")
    plt.xlabel("Scheduling Algorithm")
    plt.ylabel("Total Energy (units)")

    # Put energy value on top of each bar
    for bar, energy in zip(bars, energies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{energy:.1f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()




def read_processes_from_user() -> List[Process]:
    print("\nEnter process details.")
    while True:
        try:
            n = int(input("Number of processes: ").strip())
            if n <= 0:
                print("Please enter a positive number.\n")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.\n")

    processes: List[Process] = []
    print("Enter each process as: <PID> <arrival_time> <burst_time>")
    print("Example: A 0 4")

    for i in range(n):
        while True:
            line = input(f"Process {i+1}: ").strip()
            parts = line.split()
            if len(parts) != 3:
                print("Invalid format. Use: PID arrival burst (3 values). Try again.")
                continue
            pid = parts[0]
            try:
                arrival = int(parts[1])
                burst = int(parts[2])
            except ValueError:
                print("Arrival and burst must be integers. Try again.")
                continue
            processes.append(Process(pid, arrival, burst))
            break

    return processes


# ------------------------------
# Core Experiment Runner
# ------------------------------

def run_experiment(processes: List[Process]) -> None:
    schedulers: List[SchedulerBase] = [
        FCFSScheduler(freq=2.0),
        RRScheduler(quantum=2, freq=2.0),
        EnergyAwareScheduler(
            high_freq=2.0,
            mid_freq=1.7,
            low_freq=1.3,
            threshold_high=10,
            threshold_mid=5,
        ),
    ]

    # 1) Show input in a small table
    print_process_table(processes)

    base_metrics = None
    results = []  # list of dicts: one per algorithm

    for idx, sched in enumerate(schedulers):
        procs, segs = sched.run(processes)
        metrics = compute_metrics(procs, segs)

        if idx == 0:
            base_metrics = metrics
            energy_change = 0.0
            tat_change = 0.0
        else:
            energy_change = (
                (metrics["total_energy"] - base_metrics["total_energy"])
                / base_metrics["total_energy"]
                * 100
            )
            tat_change = (
                (metrics["avg_turnaround_time"] - base_metrics["avg_turnaround_time"])
                / base_metrics["avg_turnaround_time"]
                * 100
            )

        results.append(
            {
                "name": sched.name,
                "avg_wait": metrics["avg_waiting_time"],
                "avg_tat": metrics["avg_turnaround_time"],
                "util": metrics["cpu_utilization"] * 100,
                "energy": metrics["total_energy"],
                "d_energy": energy_change,
                "d_tat": tat_change,
            }
        )

    # 2) Compact 2D comparison table
    print("=== Algorithm Comparison Summary ===")
    print("+-------------------------------+----------+----------+----------+-----------+------------+------------+")
    print("| Algorithm                     | AvgWait  | AvgTAT   | Util(%)  | Energy    | ΔEnergy %  | ΔTAT %    |")
    print("+-------------------------------+----------+----------+----------+-----------+------------+------------+")
    for r in results:
        print(
            f"| {r['name']:<29}"
            f"| {r['avg_wait']:>8.2f} "
            f"| {r['avg_tat']:>8.2f} "
            f"| {r['util']:>8.2f} "
            f"| {r['energy']:>9.2f} "
            f"| {r['d_energy']:>10.2f} "
            f"| {r['d_tat']:>10.2f} |"
        )
    print("+-------------------------------+----------+----------+----------+-----------+------------+------------+\n")

    # 3) Very short explanation for user / viva
    print("Quick Summary:")
    print(f"- You tested {len(processes)} processes.")
    algo_names = ", ".join(r['name'] for r in results)
    print(f"- Algorithms compared: {algo_names}")

    # lowest energy & lowest TAT
    best_energy = min(results, key=lambda r: r["energy"])
    best_tat = min(results, key=lambda r: r["avg_tat"])

    print(f"- Lowest energy: {best_energy['name']} ({best_energy['energy']:.2f} units)")
    print(f"- Best (lowest) average turnaround time: {best_tat['name']} ({best_tat['avg_tat']:.2f} units)")
    print("\n")

        # Show bar chart for energy comparison
    try:
        show_energy_bar_chart(results)
    except Exception as e:
        print("Could not display energy bar chart:", e)




# ------------------------------
# Main Menu
# ------------------------------

def main() -> None:
    print("=" * 60)
    print("        ENERGY EFFICIENT CPU SCHEDULING SIMULATOR")
    print("=" * 60)
    print("This tool compares classic CPU schedulers (FCFS, Round Robin)")
    print("with an Energy-Aware DVFS-based scheduler.\n")

    while True:
        print("Menu:")
        print("  1. Use sample processes")
        print("  2. Enter custom processes")
        print("  3. Exit")
        choice = input("Choose an option (1/2/3): ").strip()

        if choice == "1":
            processes = [
                Process("A", 0, 4),
                Process("B", 1, 5),
                Process("C", 2, 3),
                Process("D", 4, 2),
            ]
            print("\nUsing sample processes.")
            run_experiment(processes)
        elif choice == "2":
            processes = read_processes_from_user()
            run_experiment(processes)
        elif choice == "3":
            print("Exiting simulator. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2 or 3.\n")


if __name__ == "__main__":
    main()
