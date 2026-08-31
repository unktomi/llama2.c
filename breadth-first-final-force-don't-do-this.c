/*
 * DO NOT USE AS THE TIMED ESCARDO INTERPRETER.
 *
 * The removed implementation queued local candidate edges until a wall-clock
 * deadline and only afterward called timed_strength_force over the completed
 * sampled tree.  Calling the queue operation observer_apply was false: the
 * composed observer had not returned p(x : b(x)), and epsilon had not selected
 * anything.  It therefore repeated the quarantined "construct first, select
 * later" organization under continuation-shaped names.
 *
 * Git history immediately before this quarantine retains the implementation
 * for audit.  This file is intentionally not an active build dependency.
 */
