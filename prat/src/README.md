# Prat: Pace Rhythm And Timing

**Prat provides an accurate, thread-safe clock and sequencer for use in creative coding projects.**
"Pace, Rhythm, And Timing" is a term used by audiophiles (people who really like expensive stereos) 
to describe a component's perceived ability to enhance rhythmic elements of recorded music. It is often described 
as an ability to induce the listener to start tapping their toes along with the sound.

This term is a source of controversy because nobody can agree on what causes it or how to measure it. Most
would characterize it as a psychoacoustic phenomenon arising from extremely accurate timing lacking jitter, 
with different downstream results in the digital and analog domains, such as accurate transient reproduction,
or something.

We named our clock module Prat as a tongue-in-cheek homage to the tension between the objective, mathematical 
foundation of music and the subjective, psychological perception of it.


**Features**
ClockService:
- Ableton Link compatible
- Grid timing accuracy even if ticks are dropped
- Messaging interface for commands and parameter changes
- Ability to set timing thread priority (tested for macOS)

Sequencer:
- A trait that allows an external module to pair with ClockService to receive beat subdivision signals.
- Roll your own methods to read data sources, map to parameters, and send OSC.

**Examples**
To start a new ClockService:
```rust
    let mut clock = ClockService::with()
        .tempo(120.0)
        .quantum(4.0)
        .ppqn(24)
        .enable_start_stop_sync()
        .enable_ticks()
        .thread_priority(47) // highest macOS thread priority, almost never drops ticks
        .debug()
        .build();
```

Subscribe to beat events:
```rust
let beat_rx = clock.subscribe_to_beats();
```

Receive beat events and trigger something on a quarter note subdivision:
```rust
while let Ok(beat_event) = beat_rx.try_recv() {
    if beat_event.subdivisions.contains(&BeatSubdivision::Quarter) {
        trigger(something);
    }
}
```





