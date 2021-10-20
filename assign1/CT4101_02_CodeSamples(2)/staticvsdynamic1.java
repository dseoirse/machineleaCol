public class staticvsdynamic1 {
	public static void main(String[] args) {
		int x = 4; // compiles ok
    x = 5;  // compiles ok
    x = 3.14159; // compiler error
    x = "Hello world!"; // compiler error
    y = "abc123"; // compiler error
	}
}
