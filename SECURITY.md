# Security

**scisolve executes model-generated code in your own Python process.** Treat it
like running a script a stranger emailed you. Don't point it at untrusted
problem statements.

There is no sandbox, no memory limit and no network block. Those absences are
deliberate and explained in the README under "What it isn't"; the short version
is that an in-process guard against code running in that same process is
bypassable, and a guard the documentation would have to lie about is worse than
a documented absence. The cell timeout uses `signal.setitimer`, so it is Unix
and main-thread only, it cannot interrupt a call that is inside compiled code,
and model code can swallow it with a bare `except Exception`.

If you want isolation, run scisolve inside a container or a VM you are willing
to lose.

## Reporting a problem

Open an issue at https://github.com/ShubhamX57/scisolve/issues. If it is
something you would rather not post publicly, say so in the issue without the
details and we will find another channel.

Please don't report "it runs arbitrary code" — that is the documented design,
not a vulnerability. A bug in how that design is described, or a way the
grounding check can be made to pass on a number the session never produced, is
worth reporting.
