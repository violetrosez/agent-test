import mysql from 'mysql2/promise';

async function main() {

    const connectionConfig = {
        host: "localhost",
        port: 3307,
        user: "root",
        password: "admin",
        multipleStatements: true,
    };

    const connection = await mysql.createConnection(connectionConfig);


    // 创建 database
    await connection.query(`CREATE DATABASE IF NOT EXISTS hello CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;`);
    await connection.query(`USE hello;`);



    // 创建好友表
    await connection.query(`
      CREATE TABLE IF NOT EXISTS friends (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(50) NOT NULL,
        gender VARCHAR(10),                -- 性别
        birth_date DATE,                   -- 出生日期
        company VARCHAR(100),              -- 公司
        title VARCHAR(100),                -- 职位
        phone VARCHAR(20),                 -- 当前手机号
        wechat VARCHAR(50)                 -- 微信号
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    `
    );

    // 插入数据
    const result = await connection.execute(`INSERT INTO friends (name, gender, birth_date, company, title, phone, wechat) VALUES (?, ?, ?, ?, ?, ?, ?);`, [
        '张三',
        '男',
        '1990-01-01',
        '公司1',
        '职位1',
        '12345678901',
        'wechat1'
    ]);

    console.log(result);

}

main();