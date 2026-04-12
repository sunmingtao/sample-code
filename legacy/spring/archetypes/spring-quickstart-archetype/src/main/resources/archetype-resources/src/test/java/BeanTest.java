package ${package};

import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.AbstractJUnit4SpringContextTests;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

import ${package}.Bean;

import static junit.framework.Assert.*;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath:context/context.xml"})
public class BeanTest extends AbstractJUnit4SpringContextTests{

	/**
	 * Just to test the context can be loaded OK
	 */
	@Test
	public void testContext(){
		Bean bean = applicationContext.getBean(Bean.class);
		assertEquals("Hello World", bean.getName());
	}
}
