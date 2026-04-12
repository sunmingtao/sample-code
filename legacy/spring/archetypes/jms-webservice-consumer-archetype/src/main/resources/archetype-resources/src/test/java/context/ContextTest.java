package ${package}.context;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

import junit.framework.Assert;

/**
 * @author 
 */
@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath:context/application-context.xml", 
		"classpath:context/test-jndi-context.xml", 
		"classpath:context/local-properties-config.xml"})
public class ContextTest{
	
	
	/**
	 * Just to test the context can be loaded OK
	 */
	@Test
	public void testContext(){
		Assert.assertTrue(true);
	}
}
