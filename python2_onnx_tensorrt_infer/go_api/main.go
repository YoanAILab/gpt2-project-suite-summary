package main

// 开发时用 go run，直接编译到内存中运行
// 发布/打包时用 go build ➔ 生成 .exe ➔ 执行。

// 导入需要用到的包：
// bytes：把 JSON 数据转成可以传输的格式
// encoding/json：做 JSON 编码（结构体 → JSON）和解码（JSON → 结构体）
// github.com/gin-gonic/gin：Gin Web 框架，负责网页路由、API 处理
// ioutil：读取 HTTP 响应体内容
// net/http：发起 HTTP 请求（比如调用 Flask 服务）
// path/filepath：处理路径，比如查找 templates/*.html
import (
	"bytes"
	"encoding/json"
	"github.com/gin-gonic/gin"
	"io/ioutil"
	"net/http"
	"path/filepath"
)

// type RequestBody struct {} ： 定义了一个叫 RequestBody 的结构体类型
// Prompt string ： 定义了一个叫 Prompt 的字段，类型是 string（字符串）
// `json:"prompt"` ： 告诉 Go 当处理 JSON 数据时，这个字段对应 JSON 里的 "prompt" 这个 key
type RequestBody struct {
	Prompt string `json:"prompt"`
}

type ResponseBody struct {
	Response string `json:"response"`
}

func main() {
	r := gin.Default() // Gin 是 Go 语言（Golang）里非常流行的一个Web框架,写 Web 后端服务用
	r.LoadHTMLGlob(filepath.Join("templates", "*.html"))
	r.Static("/static", "./static") // 映射，浏览器请求 /static/xxx.css，实际上访问的是本地的 static/xxx.css

	// 网页入口
	r.GET("/", func(c *gin.Context) {
		c.HTML(200, "index.html", nil)
	})

	// API 调用（支持页面 JS 和 Postman 调用）
	r.POST("/infer", func(c *gin.Context) {
		var req RequestBody
		if err := c.BindJSON(&req); err != nil {
			c.JSON(400, gin.H{"error": "Invalid request"}) //给前端一个响应
			return                                         // 这一次请求就结束了，但服务器继续正常跑着
		}

		// 请求 Python 后端
		reqJson, _ := json.Marshal(req)
		resp, err := http.Post("http://localhost:6006/infer", "application/json", bytes.NewBuffer(reqJson))
		if err != nil {
			c.JSON(500, gin.H{"error": "调用 Flask 服务失败", "detail": err.Error()})
			return
		}
		defer resp.Body.Close()

		body, _ := ioutil.ReadAll(resp.Body)
		var result ResponseBody
		if err := json.Unmarshal(body, &result); err != nil {
			c.JSON(500, gin.H{"error": "返回格式异常", "raw": string(body)})
			return
		}

		c.JSON(200, result)
	})

	r.Run(":8080") //http://localhost:8080/
}
