#define _GNU_SOURCE
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
static unsigned char alternate_stack[131072];
static void crash_handler(int signum, siginfo_t *info, void *context) {
    (void)info; (void)context;
    void *frames[96];
    const char marker[]="\nNATIVE_CRASH_BACKTRACE_BEGIN\n";
    write(STDERR_FILENO,marker,sizeof(marker)-1);
    int count=backtrace(frames,96);
    backtrace_symbols_fd(frames,count,STDERR_FILENO);
    const char end[]="NATIVE_CRASH_BACKTRACE_END\n";
    write(STDERR_FILENO,end,sizeof(end)-1);
    _exit(128+signum);
}
int install_native_trace(void) {
    void *warmup[2]; backtrace(warmup,2);
    stack_t stack={.ss_sp=alternate_stack,.ss_size=sizeof(alternate_stack),.ss_flags=0};
    if (sigaltstack(&stack,0)) return -1;
    struct sigaction action={0};
    action.sa_sigaction=crash_handler;
    action.sa_flags=SA_SIGINFO|SA_ONSTACK|SA_RESETHAND;
    sigemptyset(&action.sa_mask);
    if(sigaction(SIGSEGV,&action,0)) return -2;
    if(sigaction(SIGBUS,&action,0)) return -3;
    return 0;
}
