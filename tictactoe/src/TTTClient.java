import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.Scanner;

public class TTTClient {
	String anothergame;	          // the parameter of whether try another time
	   Socket socket;                 // 
	   PrintWriter printWriter;       // 
	   BufferedReader sysBuff;        // 
	   BufferedReader bufferedReader; //
	   
	   
	   /**
		  * The method can creat a new cilent and connect to the server.
		  * Then send the string user input, get the result and
		  * print the picture and the hideword.
		  *
		  * @param    String ip  
		  * @param    int    port  
		  * 
		  * @return   no 
		  */
	   public void send(String ip,int port){
	      
	      try{
	      
	        //the reader used for the user inputing from the keyboard
	         sysBuff = new BufferedReader(new InputStreamReader(System.in));
	      
	         
	         
	         // create a temporary connection to the server, while retriving the
	         // port number of the new port
	         Socket tempSocket = new Socket(ip, port);
	         BufferedReader in = new BufferedReader(new InputStreamReader
	                                 (tempSocket.getInputStream()));
	         PrintWriter out = new PrintWriter(tempSocket.getOutputStream(), true);
	         
	         System.out.println("Please enter your user name: ");
	         String userName = sysBuff.readLine();
	         
	         //transfer the user information to the server
	         out.println(userName);
	      
	         int newPort = Integer.parseInt(in.readLine());
	         System.out.println("Entering game room "+ newPort);
	         tempSocket.close();
	         in.close();
	         out.close();
	           
	         
	        	 // get a new socket with the ip and port
	         socket = new Socket(ip,newPort); 
	         
	         // the writer used for outputing the string to the server
	         printWriter = new PrintWriter(socket.getOutputStream(), true);
	         
	                   
	         // the reader used for getting the string from server
	         bufferedReader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
	            
	      
	      }catch(Exception e){
	         e.printStackTrace();
	         System.out.println("Cannot connect to the server, please contact admin: Xiaohai");
	         System.exit(0);
	      }
	      
	      
	      // continute to run the game until user decide otherwise  
	      do{
	         try{
	          	
	            
	            
	         
	         }catch(Exception e){
	            System.out.println(e);
	          
	         }
	      }
	      while(anothergame.equals("y"));
	      System.out.println("Bye-bye!");
	   }
		
	   
	   
	   public static void main(String[] args) {
		     
		      // open a scanner to get the ip and port number seperated by " "
		      Scanner sc = new Scanner(System.in);
		      System.out.println("Please input the ip and port number");
		      String arr[] = sc.nextLine().split(" ");
		      String ip = arr[0];
		      int port = Integer.parseInt(arr[1]);
		     
		      // new classs object, and call send method 
		      new TTTClient().send(ip,port);	
		      // close scanner: user --> client program
		      sc.close();
		   }
}
