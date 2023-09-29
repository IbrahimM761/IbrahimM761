import java.util.*;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.io.*;
import javax.swing.*;


public class Olypics_Logo extends JPanel{
     Circle c1 = new Circle(195, 200);
     Circle c2 = new Circle(222,225);
     Circle c3 = new Circle(252, 200);
     Circle c4 = new Circle(c2.x+60,225);
     Circle c5 = new Circle(c3.x+58,c3.y);
   

        public Olypics_Logo(){
            JFrame frame = new JFrame();
            this.setBackground(Color.GRAY);
            this.setPreferredSize(new Dimension(500,500));


            frame.add(this);
            frame.pack();
            frame.setLayout(null);
            frame.setVisible(true);
            
            }

            @Override
            protected void paintComponent(Graphics g){
                super.paintComponent(g);
                g.setColor(Color.BLUE);
                c1.draw(g);
                g.setColor(Color.YELLOW);
                c2.draw(g);
                g.setColor(Color.BLACK);
                c3.draw(g);
                g.setColor(Color.GREEN);
                c4.draw(g);
                g.setColor(Color.RED);
                c5.draw(g);
               

            }

    public class Circle{
        int x; int y; int q; int w;
        public Circle (int x, int y){
            this.x = x;
            this.y = y;
        }
        public Circle(int x,int y ,int q, int w){
            this.x = x;
            this.y = y;
            this.q = q;
            this.w = w;
        }

    public void draw(Graphics g){
        Graphics2D g2d = (Graphics2D) g;
        g2d.setStroke(new BasicStroke(4));
        g2d.drawOval(x, y, 50, 50);
        
        
       



    }


}

}
