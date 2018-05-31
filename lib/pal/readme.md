Why std::string instead of const char* for many of the low level APIs, such as the OS file APIs?
The reason is because very often, you are doing something like os::RemoveAll(path + PathSeparator + file).
If you want that to work with const char*, then you need to write 
os::RemoveAll((path + PathSeparator + file).c_str()). The only downside to using std::string, is that
if you are using a different type of string, then you've got the string alloc + copy overhead. For
functions where performance is important, we still use const char*, but for cases such as calling into
the OS, the overhead of a string alloc is minimal, so it's worth the cost.