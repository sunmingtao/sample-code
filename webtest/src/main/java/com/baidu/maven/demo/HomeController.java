package com.baidu.maven.demo;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import com.baidu.bae.api.util.BaeEnv;

@Controller
public class HomeController {
	@RequestMapping("person/{id}")
	@ResponseBody
	public Customer getById(@PathVariable String id) {
		return getCustomer(id);
	}

	@RequestMapping("db")
	@ResponseBody
	public String getDbContext() {
		String host = BaeEnv.getBaeHeader(BaeEnv.BAE_ENV_ADDR_SQL_IP);
		String port = BaeEnv.getBaeHeader(BaeEnv.BAE_ENV_ADDR_SQL_PORT);
		String username = BaeEnv.getBaeHeader(BaeEnv.BAE_ENV_AK);
		String password = BaeEnv.getBaeHeader(BaeEnv.BAE_ENV_SK);
		return "host: " + host + " port: " + port + " username: " + username
				+ "password: " + password;
	}

	public Customer getCustomer(String phoneNumber) {
		String host = BaeEnv.getBaeHeader(BaeEnv.BAE_ENV_ADDR_SQL_IP);
		String port = BaeEnv.getBaeHeader(BaeEnv.BAE_ENV_ADDR_SQL_PORT);
		String username = BaeEnv.getBaeHeader(BaeEnv.BAE_ENV_AK);
		String password = BaeEnv.getBaeHeader(BaeEnv.BAE_ENV_SK);

		String driverName = "com.mysql.jdbc.Driver";
		String dbUrl = "jdbc:mysql://";
		String serverName = host + ":" + port + "/";

		// 从平台查询应用要使用的数据库名
		String databaseName = "dLpogNtJpcAyakaaxZRp";
		String connName = dbUrl + serverName + databaseName;
		String sql = "select * from T_CUSTOMER where phoneNumber='"
				+ phoneNumber + "'";

		Connection connection = null;
		Statement stmt = null;
		ResultSet rs = null;
		Customer customer = null;
		try {
			Class.forName(driverName);
			connection = DriverManager.getConnection(connName, username,
					password);
			stmt = connection.createStatement();
			rs = stmt.executeQuery(sql);
			if (rs.next()) {
				String phone = rs.getString("phoneNumber");
				int transMade = rs.getInt("transMade");
				customer = new Customer(phone, transMade);
			}
		} catch (ClassNotFoundException ex) {
		} catch (SQLException e) {
		} finally {
			try {
				if (connection != null) {
					connection.close();
				}
			} catch (SQLException e) {
			}
		}
		return customer;
	}
}
