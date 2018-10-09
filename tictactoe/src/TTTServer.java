import java.net.ServerSocket;
import java.net.Socket;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.InetAddress;

public class TTTServer extends Thread{
	
	ServerSocket seversocket;
	Socket socket;
	int port = 4242;
	int id = 0;
	String playerName = "";
	PrintWriter out;
	BufferedReader in;
    /**
     * A class constructor that takes a port number, and uses it to create
     * a server socket that listens to clients, used for the main socket that 
     * generates thread sockets thoughout the program runtime. 
     *
     * @param: int port - 0, a randomly selected port which is not in use.
     */
   public TTTServer(int port){
      try{
         seversocket = new ServerSocket(port);
      }catch(Exception e){
         System.out.println(e);
      }
   }
   
   /** 
    * A class construtor that takes a port number, and a unique id to create
    * mark the connection with each players. 
    *
    * @param: int port - the port number 
    * @param: int id - the unique id that represent the player
    * @param: String name - the name of the player
    */
  public TTTServer(int port, int id, String name){
     this(port);
     this.id = id;
     this.playerName = name;
  }
	
   /**
    * The getLocalPort methods return the port number on the current player 
    * connection socket 
    * 
    * @param: int - the port number of the current player socket 
    */
   	public int getLocalPort(){
	   return seversocket.getLocalPort();
   	}
   	
	/**
	  * The method can creat a new sever and wait a client to connect.
	  * Receiving the guessing letter and return the result to the client.
	  * 
	  * @param    int    port  
	  * 
	  * @return   no 
	  */
   	public void run() {
   		try {
   		//a client connect to the server
            Socket clientConn = seversocket.accept();
         	
         	// get the writer using for transfer string to the client
            out = new PrintWriter (clientConn.getOutputStream (), true);
         	
         	//get the reader using for receiving the string from the client
            in = new BufferedReader( new InputStreamReader( clientConn.getInputStream()));
            
   		}catch(Exception e) {
   			
   		}
   	}
   	
   	
   	
	public void listenToPort() {
		try {
			int id = 0;
			while(true) {
				System.out.println("To join the game, connect to: " + (InetAddress.getLocalHost()+"").split("/")[1]
                        + " " + getLocalPort());
				Socket clientConn = seversocket.accept();
				BufferedReader in = new BufferedReader( new InputStreamReader( clientConn.getInputStream()));
				String userjoin = in.readLine();
				TTTServer ts = new TTTServer(0, id++, userjoin);
				System.out.println(userjoin + " transferred to gameroom: " + ts.getLocalPort() + "\n");
	            ts.start();
	            PrintWriter out = new PrintWriter (clientConn.getOutputStream(), true);
	            System.out.println("send this to client" + ts.getLocalPort());
	            out.println(ts.getLocalPort());
	            clientConn.close();    
	            in.close();
	            out.close();
			}
			
		}catch(Exception e) {
			System.out.println(e);
		}
	}
	
	public static void main(String[] args) {
		new TTTServer(0).listenToPort();
	}
}
